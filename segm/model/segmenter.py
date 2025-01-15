import torch
import torch.nn as nn
import torch.nn.functional as F

from segm.model.utils import padding, unpadding
from timm.models.layers import trunc_normal_


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.multi_token_mlp = nn.Linear(encoder.d_model, encoder.d_model)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def cal_loss(self, pos, neg, dot):
        logits = dot

        neg_logits = torch.exp(logits) * neg
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        # print('exp_logits ', has_inf_or_nan(exp_logits))
        log_prob = logits - torch.log(exp_logits + neg_logits)
        # print('log_prob ', has_inf_or_nan(log_prob))

        mean_log_prob_pos = (pos * log_prob).sum(1) / pos.sum(1)  # normalize by positives
        # print('\npositives: {} \nnegatives {}'.format(pos.sum(1), neg.sum(1)))
        # print('mean_log_prob_pos ', has_inf_or_nan(mean_log_prob_pos))
        loss = - mean_log_prob_pos
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.mean()

        return loss


    def get_loss(self, feats, labels, multi_cls):
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)  # L2 normalization
        num_anchors, views_per_anchor, c = feats.shape  # get T, V, C
        labels = labels.contiguous().view(-1, 1)  # labels are T-1

        # print( f'rank: {get_rank()} -- classes {num_anchors} v_per_class {views_per_anchor} total_anchors = {num_anchors * views_per_anchor}')
        # feats_flat = torch.cat(torch.unbind(feats, dim=1), dim=0)  # feats_flat is V*T-C
        # dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(feats_flat, 0, 1)), self.temperature)
        # # dot_product # V*T-C @ C-V*T = V*T-V*T
        #
        # mask, pos_mask, neg_mask = self.get_masks(labels, num_anchors, views_per_anchor)
        # loss = self.compute(pos_mask, neg_mask, dot_product)
        # print(loss)

        # modifying to more intuitive version
        labels_ = labels.repeat(1, views_per_anchor)  # labels are T-V
        labels_ = labels_.view(-1, 1)  # labels are T*V-1
        feats_flat = feats.contiguous().view(-1, c)  # feats_flat is T*V-C
        multi_cls = torch.nn.functional.normalize(multi_cls.clone(), p=2, dim=-1)

        temperature = 0.1
        dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(multi_cls.squeeze(0), 0, 1)), temperature)

        anchor_label = torch.arange(0, multi_cls.shape[1]).to(labels_.device)

        pos_mask = torch.eq(labels_, anchor_label).float()
        neg_mask = 1 - pos_mask

        loss = self.cal_loss(pos_mask, neg_mask, dot_product)

        return loss

    def global_loss(self, seg_gt, feats, multi_cls):
        patch_gt = F.interpolate(seg_gt.unsqueeze(1).float(),
                                               (seg_gt.shape[-2] // self.patch_size, seg_gt.shape[-1] // self.patch_size), mode='nearest')

        patch_gt_flat = patch_gt.flatten(1)

        classes_ids = torch.arange(start=0, end=self.n_cls, step=1, device=patch_gt_flat.device)
        compare = patch_gt_flat.unsqueeze(-1) == classes_ids.unsqueeze(0).unsqueeze(0)  # n, hw, 1 == 1, 1, n_c => n,hw,n_c
        cls_counts = compare.sum(1)

        present_inds = torch.where(cls_counts >= 10)  # ([0,...,n-1], [prese   nt class ids])
        batch_inds, cls_in_batch = present_inds

        min_views = torch.min(cls_counts[present_inds])
        total_cls = cls_in_batch.shape[0]
        c = feats.shape[-1]

        views_per_class = min_views
        sampled_features = torch.zeros((total_cls, views_per_class, c), dtype=torch.float).cuda()
        sampled_labels = torch.zeros(total_cls, dtype=torch.float).cuda()

        for i in range(total_cls):
            indices_from_cl_fast = compare[batch_inds[i], :, cls_in_batch[i]].nonzero().squeeze()
            # indices_from_cl = (dominant_classes[batch_inds[i]] == cls_in_batch[i]).nonzero().squeeze()
            random_permutation = torch.randperm(indices_from_cl_fast.shape[0]).cuda()
            sampled_indices_from_cl = indices_from_cl_fast[random_permutation[:views_per_class]]
            sampled_features[i] = feats[batch_inds[i], sampled_indices_from_cl, :]
            sampled_labels[i] = cls_in_batch[i]


        global_loss = self.get_loss(sampled_features, sampled_labels, multi_cls)

        return global_loss

    def forward(self, im, seg_gt, is_train):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        # num_extra_tokens = 1 + self.encoder.distilled
        # x = x[:, num_extra_tokens:]

        multi_cls, x = x[:, :self.n_cls, :], x[:, self.n_cls:, :]

        multi_cls = multi_cls.mean(dim=0, keepdim=True)

        global_loss = 0
        if is_train:
            global_loss = self.global_loss(seg_gt, x, multi_cls)

        multi_cls = self.multi_token_mlp(multi_cls)

        masks = self.decoder(x, multi_cls, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks, global_loss

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)



