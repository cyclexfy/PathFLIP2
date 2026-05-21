import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math

try:
    import torch.distributed.nn
    from torch import distributed as dist
    from .utils.dist_utils import concat_all_gather_with_grad

    has_distributed = True
except ImportError:
    has_distributed = False
    
    def concat_all_gather_with_grad(tensor):
        return tensor

def create_loss(args):

    if args.use_fine_gained_loss:
        if hasattr(args, 'use_soft_topk') and args.use_soft_topk:
            return SoftTopKLoss(
                num_cap_per_img=args.num_text_samples,
                top_k=args.top_k if hasattr(args, 'top_k') else 64,
                temperature=args.topk_temperature if hasattr(args, 'topk_temperature') else 0.5,
            )
        else:
            return FlairLoss(
                num_cap_per_img=args.num_text_samples,
                added_mps_loss=args.add_mps_loss if hasattr(args, 'add_mps_loss') else False,
            )
    else:
        raise NotImplementedError("Loss function for the given configuration is not implemented.")

def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)



def get_multi_positive_mps(target, k):
    """
    :param target: tensor of shape (b, b*k), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b*k), for each row i, the col [i*k, (i+1)*k] should be ones
    """
    for i in range(target.shape[0]):
        target[i, i * k:(i + 1) * k] = 1
    return target



def get_multi_positive_tcs(target, k):
    """
    :param target: tensor of shape (b, b+k-1), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b+k-1), for each row i, the col [i, i+k) should be ones
    """
    for i in range(target.shape[0]):
        target[i, i: i + k] = 1
    return target

def get_mps_logits(image_features, text_features, logit_scale, logit_bias=None):
    """
    image_features: (B, D)
    text_features: (B*K, D)
    """
    logits = logit_scale * image_features @ text_features.T  # if multi-cap: (B, B*K)
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_mps_ground_truth(device, dtype, target_shape, negative_only=False,
                                        num_captions=4):
    dim0, dim1 = target_shape  # (B, B*K)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B*K)
    if not negative_only:
        labels = get_multi_positive_mps(target=labels, k=num_captions)
    return labels

def get_intra_logits(image_features, text_features, logit_scale, logit_bias=None):
    """
    image_features: (B, K, D),
    text_features: (B, K, D).
    Target: (B, K, K)
    """
    logits = logit_scale * torch.einsum('bkd,bjd->bkj', image_features, text_features)
    # logits = logit_scale * image_features @ text_features.T  
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_tcs_ground_truth(device, dtype, target_shape, negative_only=False, num_captions=4):
    dim0, dim1 = target_shape  # (B, B+K-1)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B+K-1)
    if not negative_only:
        labels = get_multi_positive_tcs(target=labels, k=num_captions)
    return labels

def get_tcs_logits(features_0, features_1, logit_scale, logit_bias=None):
    logits = logit_scale * torch.einsum('bij,bij->bi', features_0, features_1)
    if logit_bias is not None:
        logits += logit_bias
    return logits

### FlairLoss
class FlairLoss(nn.Module):
    def __init__(
            self,
            bidir=True,
            num_cap_per_img=8, # num_sampled_captions
            added_mps_loss=False,
    ):
        super().__init__()
        self.bidir = bidir

        self.num_cap_per_img = num_cap_per_img
        self.added_mps_loss = added_mps_loss

    def _loss_with_attn_pool(self, image_features, image_tokens, text_features, logit_scale,
                             logit_bias=None, negative_only=False, visual_proj=None, g_text_features=None, attn_mask=None, output_attn_weights=False):

        if output_attn_weights:
            local_image_features, attn_weights= visual_proj(text_features, image_tokens, image_tokens, attn_mask=attn_mask, output_attn_weights=output_attn_weights)  # (B, B+K-1, D)
        else:    
            local_image_features = visual_proj(text_features, image_tokens, image_tokens, attn_mask=attn_mask, output_attn_weights=output_attn_weights)  # (B, B+K-1, D)

        local_image_features = F.normalize(local_image_features, dim=-1)
        global_text_features = F.normalize(text_features, dim=-1)

        i2t_logits = get_tcs_logits(local_image_features, global_text_features, logit_scale, logit_bias)

        i2t_labels = get_tcs_ground_truth(device=text_features.device,
                                        dtype=text_features.dtype,
                                        target_shape=i2t_logits.size(),
                                        negative_only=negative_only,
                                        num_captions=self.num_cap_per_img)

        batch_size = i2t_logits.size(0)
        num_positives = batch_size * self.num_cap_per_img

        # tcs_loss = -F.logsigmoid(i2t_labels * i2t_logits).sum() / text_features.shape[1] # text-conditioned sigmoid loss
        tcs_loss = -F.logsigmoid(i2t_labels * i2t_logits).sum() / num_positives

        if self.added_mps_loss:
            g_image_features = F.normalize(image_features, dim=-1)  #(B, D)
            g_text_features = F.normalize(g_text_features, dim=-1)  #(B*K, D)
            mps_logits = get_mps_logits(image_features=g_image_features, text_features=g_text_features,
                                                  logit_scale=logit_scale, logit_bias=logit_bias) #(B, B*K)
            g2g_labels = get_mps_ground_truth(device=g_text_features.device,
                                              dtype=g_text_features.dtype,
                                              target_shape=mps_logits.size(),
                                              negative_only=negative_only,
                                              num_captions=self.num_cap_per_img)
            mps_loss = -F.logsigmoid(g2g_labels * mps_logits).sum() / g_text_features.shape[0] # multi-positive sigmoid loss

            loss = (tcs_loss + mps_loss) / 2
        else:
            loss = tcs_loss

        if output_attn_weights:
            return loss, attn_weights
        else:
            return loss

    def forward(
            self,
            image_features, 
            text_features, 
            logit_scale, 
            logit_bias, 
            image_tokens,
            visual_proj=None, 
            output_dict=False, 
            attn_mask=None, 
            output_attn_weights=False,
            rank=0,
            world_size=1,
        ):
        '''
        expected shape: text_features: (B*K, D), image_embeddings: (B, L, D)
        '''

        batch_size = image_tokens.shape[0]
        
        expected_text_len = batch_size * self.num_cap_per_img
        actual_text_len = text_features.shape[0]
        
        if expected_text_len != actual_text_len:
            raise ValueError(
                f"Text feature count mismatch: actual {actual_text_len}, "
                f"expected B * K ({batch_size} * {self.num_cap_per_img} = {expected_text_len}). "
                "Check whether num_cap_per_img is configured correctly."
            )

        if self.added_mps_loss:
            g_text_features = text_features  # (B*K, D)
        else:
            g_text_features = None
        

        # We don't change the shape of image tokens anywhere before the loss function.
        batch_size = image_tokens.shape[0]
        num_captions = self.num_cap_per_img
        caption_indices = torch.arange(batch_size * num_captions).view(batch_size, num_captions).to(
            text_features.device)
        
        text_features = downsample_text_features(text_features=text_features, batch_size=batch_size,
                                                 caption_indices=caption_indices,
                                                 num_captions=num_captions)

        loss_out = self._loss_with_attn_pool(image_features=image_features,
                                         image_tokens=image_tokens,
                                         text_features=text_features,
                                         visual_proj=visual_proj,
                                         logit_scale=logit_scale,
                                         logit_bias=logit_bias,
                                         g_text_features=g_text_features,
                                         attn_mask=attn_mask,
                                         output_attn_weights=output_attn_weights)
        if output_attn_weights:
            loss, attn_weights = loss_out
            attn_weights_out = [] 
            for i in range(text_features.shape[0]):
                attn_weights_out.append(attn_weights[i, i: i + self.num_cap_per_img])
            attn_weights_out = torch.stack(attn_weights_out, dim=0)
        else:
            loss = loss_out

        if world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (rank + 1) % world_size
            left_rank = (rank - 1 + world_size) % world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features_to_left = g_text_features

                num_bidir, remainder = divmod(world_size - 1, 2)

                g_text_features_recv = None  # predefine it to be None

                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            g_text_features_to_left,
                            g_text_features_to_right,
                        )
                        for j in range(len(text_features_recv)):
                            loss += self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=text_features_recv[j],
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=g_text_features_recv[j]
                            )
                    else:
                        for f in text_features_recv:
                            loss += self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=f,
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=None)
                    text_features_to_left, text_features_to_right = text_features_recv
                    if self.added_mps_loss:
                        g_text_features_to_left, g_text_features_to_right = g_text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                        loss += self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=g_text_features_recv
                        )
                    else:
                        loss += self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=None)
            else:
                text_features_to_right = text_features
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features

                for i in range(world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    if self.added_mps_loss:
                        g_text_features_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                    else:
                        g_text_features_from_left = None

                    loss += self._loss_with_attn_pool(
                        image_features=image_features,
                        image_tokens=image_tokens,
                        text_features=text_features_from_left,
                        visual_proj=visual_proj,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        negative_only=True,
                        g_text_features=g_text_features_from_left)

                    text_features_to_right = text_features_from_left
                    
        # loss = loss / world_size

        if output_attn_weights:
            return {"contrastive_loss": loss, "attn_weights": attn_weights_out} if output_dict else loss, attn_weights_out
        else:
            return {"contrastive_loss": loss} if output_dict else loss
    
def soft_topk_selection(text_features, patch_features, top_k, temperature=1.0):
    """
    Soft Top-Katch
    
    Args:
        text_features: (B, K, D) or (B*K, D)
        patch_features: (B, N, D) - atch
        top_k: int - op-k
        temperature: float - softmax temperature
    
    Returns:
        aggregated_features: (B, K, D) - 
        selection_weights: (B, K, N) - atch
    """
    if text_features.dim() == 2:
        B_K, D = text_features.shape
        K = 1
        B = B_K
        text_features = text_features.unsqueeze(1)  # (B, 1, D)
    else:
        B, K, D = text_features.shape
    
    B, N, D = patch_features.shape
    
    text_normalized = F.normalize(text_features, dim=-1)  # (B, K, D)
    patch_normalized = F.normalize(patch_features, dim=-1)  # (B, N, D)
    
    similarity = torch.einsum('bkd,bnd->bkn', text_normalized, patch_normalized)  # (B, K, N)
    
    selection_weights = F.softmax(similarity / temperature, dim=-1)  # (B, K, N)
    
    topk_weights, topk_indices = torch.topk(selection_weights, top_k, dim=-1)  # (B, K, top_k)
    
    mask = torch.zeros_like(selection_weights)  # (B, K, N)
    mask.scatter_(-1, topk_indices, topk_weights)
    
    mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    mask = mask / mask_sum  # (B, K, N)
    
    aggregated_features = torch.einsum('bkn,bnd->bkd', mask, patch_features)  # (B, K, D)
    
    return aggregated_features, selection_weights

def downsample_text_features(text_features, batch_size, caption_indices, num_captions):
    device = text_features.device
    own_caption_indices = caption_indices  # Shape: (B, K)

    mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)

    other_image_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(batch_size, batch_size)
    other_image_indices = other_image_indices[mask].view(batch_size, batch_size - 1)
    random_offsets = torch.randint(0, num_captions, (batch_size, batch_size - 1), device=device)  # (B, B-1)
    other_caption_indices = caption_indices[other_image_indices, random_offsets]  # sampled indices (B, B-1)

    combined_indices = torch.cat([own_caption_indices, other_caption_indices], dim=1) # (B, K+B-1)
    combined_indices, _ = combined_indices.sort(dim=1)
    flat_combined_indices = combined_indices.view(-1)  # flatten to take the text_features out

    flat_combined_indices = flat_combined_indices.to(device=text_features.device, dtype=torch.long)

    downsampled_text_features = text_features[flat_combined_indices] # (B*(K+B-1), D)

    embed_dim = text_features.shape[-1]  # Reshape to (B, K + B - 1, D)
    downsampled_text_features = downsampled_text_features.view(batch_size, num_captions + batch_size - 1, embed_dim)
    return downsampled_text_features


class SoftTopKLoss(nn.Module):
    def __init__(
            self,
            num_cap_per_img=8,
            top_k=64,
            temperature=0.5,
    ):
        super().__init__()
        self.num_cap_per_img = num_cap_per_img
        self.top_k = top_k
        self.temperature = temperature

    def _compute_soft_topk_features(self, image_tokens, text_features):
        B, K, D = text_features.shape
        B, N, D_patch = image_tokens.shape
        
        text_normalized = F.normalize(text_features, dim=-1)
        patch_normalized = F.normalize(image_tokens, dim=-1)
        
        similarity = torch.einsum('bkd,bnd->bkn', text_normalized, patch_normalized)
        
        selection_weights = F.softmax(similarity / self.temperature, dim=-1)
        
        topk_weights, topk_indices = torch.topk(selection_weights, self.top_k, dim=-1)
        
        mask = torch.zeros_like(selection_weights)
        mask.scatter_(-1, topk_indices, topk_weights)
        
        mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        mask = mask / mask_sum
        
        local_image_features = torch.einsum('bkn,bnd->bkd', mask, image_tokens)
        
        return local_image_features, selection_weights

    def forward(
            self,
            image_features, 
            text_features, 
            logit_scale, 
            logit_bias, 
            image_tokens,
            output_dict=False, 
            output_weights=False,
            rank=0,
            world_size=1,
        ):
        batch_size = image_tokens.shape[0]
        
        expected_text_len = batch_size * self.num_cap_per_img
        actual_text_len = text_features.shape[0]
        
        if expected_text_len != actual_text_len:
            raise ValueError(f"Text feature count mismatch: got {actual_text_len}, expected {expected_text_len}")

        text_features = text_features.reshape(batch_size, self.num_cap_per_img, -1)
        
        local_image_features, selection_weights = self._compute_soft_topk_features(image_tokens, text_features)
        
        local_image_features = F.normalize(local_image_features, dim=-1)
        text_features_normalized = F.normalize(text_features, dim=-1)
        
        if world_size > 1:
            image_feats_flat = local_image_features.reshape(batch_size * self.num_cap_per_img, -1)
            text_feats_flat = text_features_normalized.reshape(batch_size * self.num_cap_per_img, -1)
            
            image_feats_all = concat_all_gather_with_grad(image_feats_flat)
            text_feats_all = concat_all_gather_with_grad(text_feats_flat)
            
            total_batch = batch_size * world_size
            total_samples = total_batch * self.num_cap_per_img
            
            logits_i2t = logit_scale * image_feats_flat @ text_feats_all.T
            logits_t2i = logit_scale * text_feats_flat @ image_feats_all.T
            if logit_bias is not None:
                logits_i2t += logit_bias
                logits_t2i += logit_bias
            
            pos_start = rank * batch_size * self.num_cap_per_img
            pos_end = pos_start + batch_size * self.num_cap_per_img
            pos_indices = torch.arange(pos_start, pos_end, device=image_feats_flat.device)
            
            labels_i2t = torch.full_like(logits_i2t, -1.0)
            labels_t2i = torch.full_like(logits_t2i, -1.0)
            labels_i2t[:, pos_indices] = 1.0
            labels_t2i[:, pos_indices] = 1.0
            
            loss_i2t = -F.logsigmoid(labels_i2t * logits_i2t).mean()
            loss_t2i = -F.logsigmoid(labels_t2i * logits_t2i).mean()
            
            loss = (loss_i2t + loss_t2i) / 2
        else:
            all_image_features = local_image_features.reshape(batch_size * self.num_cap_per_img, -1)
            all_text_features = text_features_normalized.reshape(batch_size * self.num_cap_per_img, -1)
            
            logits = logit_scale * all_image_features @ all_text_features.T
            if logit_bias is not None:
                logits += logit_bias
            
            labels = torch.eye(batch_size * self.num_cap_per_img, device=logits.device)
            labels = 2 * labels - 1
            
            loss_i2t = -F.logsigmoid(labels * logits).mean()
            loss_t2i = -F.logsigmoid(labels * logits.T).mean()
            
            loss = (loss_i2t + loss_t2i) / 2

        if output_weights:
            return {"contrastive_loss": loss, "weights": selection_weights} if output_dict else loss, selection_weights
        else:
            return {"contrastive_loss": loss} if output_dict else loss
