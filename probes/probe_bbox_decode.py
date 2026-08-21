"""pi0.5 native bbox-retention probe (BEHAVIOR-2026 W2).

Question: does a pi05 checkpoint (fine-tuned and/or base) still retain PaliGemma's
pretrained detect-style grounding ("detect radio\n" -> "<locYYYY>"*4 + label)?
openpi ships no autoregressive text-decode path (GitHub issue #664), so we build one:

  - load raw params via openpi.models.model.restore_params (NOT the policy wrapper)
  - rebuild the pure-linen SigLIP + Gemma modules exactly as pi0.Pi0.__init__ does
  - embed prefix (image tokens + BOS + prompt + "\n") exactly like Pi0.embed_prefix
    (bidirectional attention over the prefix == PaliGemma's native prefix-LM setup)
  - greedy AR decode over the PaliGemma expert (expert 0) with a FIXED-SIZE KV cache
    (zero-padded, masked; new k/v written via dynamic_update_slice -> single compile)
  - logits = final_normed_hidden @ embedding_table.T (big_vision convention)
  - VERIFICATION: teacher-forced single full forward pass over prefix + all generated
    tokens must reproduce the same greedy argmax at every step (guards against
    KV-cache bugs masquerading as "model lost grounding"). Greedy argmax == temp 0
    == top-k 1 by construction.

Run (on the box):
  cd /workspace/openpi && uv run python /workspace/probes/probe_bbox_decode.py \
      --params /root/baseline_ckpt/pi05_turn_on_the_radio/params \
      --frames /root/probe_frames --out /root/probe_out/finetuned --tag finetuned
"""

import argparse
import dataclasses
import json
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw
import sentencepiece

import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models.model import restore_params
from openpi.models.pi0 import make_attn_mask

MAX_NEW = 48  # total generated tokens (incl. the first one predicted from prefill)
IMG_TOKENS = 256  # 224/14 -> 16x16 for So400m/14
LOC_RE = re.compile(r"^<loc(\d{4})>$")

PROMPTS = [
    "detect radio",
    "detect radio ; table ; couch",
    "answer en where is the radio?",
    "caption en",
]


def load_tokenizer(path: str) -> sentencepiece.SentencePieceProcessor:
    with open(path, "rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def tokenize_prompt(sp, text: str) -> list[int]:
    # Mirror openpi PaligemmaTokenizer pi0-format: BOS + prompt + separately-encoded "\n"
    return sp.encode(text, add_bos=True) + sp.encode("\n")


def load_frames(frames_dirs: str) -> tuple[np.ndarray, list[str], list[tuple[int, int]], dict]:
    """frames_dirs: comma-separated dirs; frame names are prefixed with the dir basename."""
    imgs, names, sizes, paths_by_name = [], [], [], {}
    for d in frames_dirs.split(","):
        paths = sorted(pathlib.Path(d).glob("*.png"))
        assert paths, f"no PNG frames in {d}"
        prefix = pathlib.Path(d).name
        for p in paths:
            im = Image.open(p).convert("RGB")
            sizes.append(im.size)  # (W, H)
            # PaliGemma-native STRETCH resize to 224x224 (pretraining distribution for
            # detect), not openpi's resize_with_pad. Noted in the report.
            im224 = im.resize((224, 224), Image.BICUBIC)
            imgs.append(np.asarray(im224, dtype=np.float32) / 127.5 - 1.0)
            name = f"{prefix}__{p.stem}"
            names.append(name)
            paths_by_name[name] = str(p)
    return np.stack(imgs), names, sizes, paths_by_name


@dataclasses.dataclass
class Decoder:
    llm: object
    llm_params: dict
    embed_table: jnp.ndarray  # [vocab, width] (bf16)
    txt_len: int  # padded text length

    def __post_init__(self):
        self._prefill_j = jax.jit(self._prefill)
        self._decode_j = jax.jit(self._decode)
        self._verify_j = jax.jit(self._verify)

    # ---- pieces -------------------------------------------------------------
    def _emb_txt(self, tokens):  # [b,t] int32 -> [b,t,d]
        return self.llm.apply({"params": self.llm_params}, tokens, method="embed")

    def _logits(self, hidden):  # [..., d] -> [..., vocab] in f32
        return jnp.einsum(
            "...d,vd->...v", hidden, self.embed_table, preferred_element_type=jnp.float32
        )

    def _fwd(self, emb, positions, mask, kv_cache=None):
        (out0, _), kv = self.llm.apply(
            {"params": self.llm_params}, [emb, None], positions, mask, kv_cache=kv_cache
        )
        return out0, kv

    # ---- prefill: bidirectional prefix (image+text), like Pi0.embed_prefix ---
    def _prefill(self, img_emb, txt_tokens, txt_mask):
        b = txt_tokens.shape[0]
        txt_emb = self._emb_txt(txt_tokens)
        if img_emb is not None:
            emb = jnp.concatenate([img_emb, txt_emb], axis=1)
            input_mask = jnp.concatenate(
                [jnp.ones((b, img_emb.shape[1]), bool), txt_mask], axis=1
            )
        else:
            emb, input_mask = txt_emb, txt_mask
        ar_mask = jnp.zeros(emb.shape[1], bool)  # fully bidirectional prefix
        attn = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        out0, kv = self._fwd(emb, positions, attn, None)
        n_valid = jnp.sum(input_mask, axis=1)  # [b]
        last_idx = n_valid - 1
        last_hidden = jnp.take_along_axis(out0, last_idx[:, None, None], axis=1)[:, 0]
        logits0 = self._logits(last_hidden)
        tok0 = jnp.argmax(logits0, axis=-1).astype(jnp.int32)
        top2 = jax.lax.top_k(logits0, 2)[0]
        return kv, input_mask, n_valid, tok0, (top2[:, 0] - top2[:, 1])

    # ---- greedy decode with fixed-size zero-padded KV cache ------------------
    def _decode(self, kv, prefix_valid, n_valid, tok0):
        k, v = kv  # [l,b,P,kh,h]
        b, p = k.shape[1], k.shape[2]
        pad = [(0, 0), (0, 0), (0, MAX_NEW), (0, 0), (0, 0)]
        bufk, bufv = jnp.pad(k, pad), jnp.pad(v, pad)

        def step(carry, t):
            bufk, bufv, tok = carry
            emb = self._emb_txt(tok[:, None])  # [b,1,d]
            positions = (n_valid + t)[:, None].astype(jnp.int32)  # [b,1]
            # cache validity: prefix valid slots + already-written decode slots [p, p+t)
            dec_valid = (jnp.arange(MAX_NEW) < t)[None, :]  # [1,48]
            valid = jnp.concatenate(
                [prefix_valid, jnp.broadcast_to(dec_valid, (b, MAX_NEW))], axis=1
            )  # [b, cache_size]
            mask = jnp.concatenate(
                [valid[:, None, :], jnp.ones((b, 1, 1), bool)], axis=-1
            )  # [b,1,cache_size+1] (self always attends)
            out0, (kf, vf) = self._fwd(emb, positions, mask, kv_cache=(bufk, bufv))
            newk, newv = kf[:, :, -1:], vf[:, :, -1:]
            bufk = jax.lax.dynamic_update_slice_in_dim(bufk, newk, p + t, axis=2)
            bufv = jax.lax.dynamic_update_slice_in_dim(bufv, newv, p + t, axis=2)
            logits = self._logits(out0[:, -1])  # [b,vocab] f32
            nxt = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            top2 = jax.lax.top_k(logits, 2)[0]
            return (bufk, bufv, nxt), (nxt, top2[:, 0] - top2[:, 1])

        (_, _, _), (toks, gaps) = jax.lax.scan(
            step, (bufk, bufv, tok0), jnp.arange(MAX_NEW - 1)
        )
        # generated sequence: tok0 then toks (steps 0..46 produce tokens 1..47)
        gen = jnp.concatenate([tok0[:, None], toks.T.astype(jnp.int32)], axis=1)  # [b,48]
        return gen, gaps.T  # gaps for tokens 1..47

    # ---- verification: single teacher-forced full forward pass ---------------
    def _verify(self, img_emb, txt_tokens, txt_mask, gen):
        b = txt_tokens.shape[0]
        txt_emb = self._emb_txt(txt_tokens)
        gen_emb = self._emb_txt(gen)
        parts = [txt_emb, gen_emb] if img_emb is None else [img_emb, txt_emb, gen_emb]
        emb = jnp.concatenate(parts, axis=1)
        n_img = 0 if img_emb is None else img_emb.shape[1]
        prefix_len = n_img + txt_tokens.shape[1]
        input_mask = jnp.concatenate(
            [jnp.ones((b, n_img), bool), txt_mask, jnp.ones((b, MAX_NEW), bool)], axis=1
        )
        ar_mask = jnp.concatenate(
            [jnp.zeros(prefix_len, bool), jnp.ones(MAX_NEW, bool)]
        )  # prefix bidirectional, generated causal
        attn = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        out0, _ = self._fwd(emb, positions, attn, None)
        n_valid = jnp.sum(input_mask[:, :prefix_len], axis=1)
        # hidden states predicting gen[i]: last valid prefix token for i=0,
        # generated token i-1 (abs index prefix_len+i-1) for i>=1
        idx0 = (n_valid - 1)[:, None]  # [b,1]
        idx_rest = prefix_len + jnp.arange(MAX_NEW - 1)[None, :] + jnp.zeros((b, 1), jnp.int32)
        idx = jnp.concatenate([idx0, idx_rest], axis=1)  # [b,48]
        hid = jnp.take_along_axis(out0, idx[:, :, None], axis=1)  # [b,48,d]
        logits = self._logits(hid)  # [b,48,vocab]
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        lp = jnp.take_along_axis(logits, pred[..., None], axis=-1)[..., 0]
        lg = jnp.take_along_axis(logits, gen[..., None].astype(jnp.int32), axis=-1)[..., 0]
        return pred, lp - lg  # gap>=0; ==0 where pred==gen

    # ---- public -------------------------------------------------------------
    def generate(self, img_emb, txt_tokens, txt_mask):
        kv, prefix_valid, n_valid, tok0, gap0 = self._prefill_j(img_emb, txt_tokens, txt_mask)
        gen, gaps = self._decode_j(kv, prefix_valid, n_valid, tok0)
        pred, vgap = self._verify_j(img_emb, txt_tokens, txt_mask, gen)
        gen, pred, vgap = np.asarray(gen), np.asarray(pred), np.asarray(vgap)
        mism = gen != pred
        verify = {
            "n_mismatch": int(mism.sum()),
            "n_total": int(gen.size),
            "max_logit_gap_at_mismatch": float(vgap[mism].max()) if mism.any() else 0.0,
            "mismatch_positions": np.argwhere(mism).tolist()[:20],
        }
        return gen, verify


def pieces_of(sp, ids):
    return [sp.id_to_piece(int(i)) for i in ids]


def cut_at_eos(sp, ids):
    ids = list(map(int, ids))
    eos = sp.eos_id()
    return ids[: ids.index(eos)] if eos in ids else ids


def parse_boxes(pieces, orig_w, orig_h):
    """Parse PaliGemma detect output: <locY1><locX1><locY2><locX2> label [; ...]."""
    boxes, i = [], 0
    while i < len(pieces):
        m = [LOC_RE.match(pieces[i + j]) if i + j < len(pieces) else None for j in range(4)]
        if all(m):
            y1, x1, y2, x2 = (int(mm.group(1)) for mm in m)
            i += 4
            label = []
            while i < len(pieces) and not LOC_RE.match(pieces[i]) and pieces[i].strip("▁") != ";":
                label.append(pieces[i])
                i += 1
            if i < len(pieces) and pieces[i].strip("▁") == ";":
                i += 1
            boxes.append(
                {
                    "label": "".join(label).replace("▁", " ").strip(),
                    "yxyx_1024": [y1, x1, y2, x2],
                    "xyxy_px": [
                        x1 / 1024 * orig_w,
                        y1 / 1024 * orig_h,
                        x2 / 1024 * orig_w,
                        y2 / 1024 * orig_h,
                    ],
                }
            )
        else:
            i += 1
    return boxes


def draw_boxes(frame_path, boxes, out_path):
    im = Image.open(frame_path).convert("RGB")
    d = ImageDraw.Draw(im)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
    for bi, b in enumerate(boxes):
        x1, y1, x2, y2 = b["xyxy_px"]
        c = colors[bi % len(colors)]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        d.text((x1 + 3, max(0, y1 - 12)), b["label"], fill=c)
    im.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="ckpt")
    ap.add_argument(
        "--tokenizer", default="/root/.cache/openpi/big_vision/paligemma_tokenizer.model"
    )
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = load_tokenizer(args.tokenizer)
    print(f"tokenizer: vocab={sp.vocab_size()} bos={sp.bos_id()} eos={sp.eos_id()}")

    print(f"loading params from {args.params} (bf16)...")
    params = restore_params(args.params, dtype=jnp.bfloat16)
    print("top-level params keys:", list(params.keys()))
    pg = params["PaliGemma"]

    paligemma_config = _gemma.get_config("gemma_2b")
    action_expert_config = _gemma.get_config("gemma_300m")
    llm = _gemma.Module(
        configs=[paligemma_config, action_expert_config], embed_dtype="bfloat16", adarms=True
    )
    img_mod = _siglip.Module(
        num_classes=paligemma_config.width,
        variant="So400m/14",
        pool_type="none",
        scan=True,
        dtype_mm="bfloat16",
    )
    embed_table = pg["llm"]["embedder"]["input_embedding"]

    frames, names, sizes, paths_by_name = load_frames(args.frames)
    print(f"frames: {len(names)} {names}")
    img_emb, _ = jax.jit(lambda x: img_mod.apply({"params": pg["img"]}, x, train=False))(
        jnp.asarray(frames)
    )
    print("img_emb:", img_emb.shape, img_emb.dtype)

    tok_lists = [tokenize_prompt(sp, p) for p in PROMPTS]
    txt_len = max(len(t) for t in tok_lists)
    print("prompt token ids:", {p: t for p, t in zip(PROMPTS, tok_lists)})

    dec = Decoder(llm=llm, llm_params=pg["llm"], embed_table=embed_table, txt_len=txt_len)

    results = {"tag": args.tag, "params": args.params, "prompts": {}, "lm_sanity": {}}

    # ---- text-only LM sanity (no image): trivially-known continuations ------
    sanity_strs = ["The capital of France is", "1 2 3 4 5 6 7 8 9"]
    sanity_ids = [sp.encode(s, add_bos=True) for s in sanity_strs]
    sanity_len = max(len(i) for i in sanity_ids)  # shared shape -> one compile
    for sanity, ids in zip(sanity_strs, sanity_ids):
        toks = np.zeros((1, sanity_len), np.int32)
        mask = np.zeros((1, sanity_len), bool)
        toks[0, : len(ids)], mask[0, : len(ids)] = ids, True
        gen, verify = dec.generate(None, jnp.asarray(toks), jnp.asarray(mask))
        ids_cut = cut_at_eos(sp, gen[0])
        results["lm_sanity"][sanity] = {
            "decoded": sp.decode(ids_cut),
            "pieces": pieces_of(sp, gen[0][:12]),
            "verify": verify,
        }
        print(f"[lm-sanity] {sanity!r} -> {results['lm_sanity'][sanity]['decoded']!r} "
              f"verify={verify['n_mismatch']}/{verify['n_total']} "
              f"(max gap {verify['max_logit_gap_at_mismatch']:.4f})")

    # ---- image prompts -------------------------------------------------------
    b = len(names)
    for prompt, ids in zip(PROMPTS, tok_lists):
        toks = np.zeros((b, txt_len), np.int32)
        mask = np.zeros((b, txt_len), bool)
        toks[:, : len(ids)], mask[:, : len(ids)] = ids, True
        gen, verify = dec.generate(img_emb, jnp.asarray(toks), jnp.asarray(mask))
        pr = {"token_ids": ids, "verify": verify, "frames": {}}
        print(f"\n=== PROMPT {prompt!r}  verify mismatches "
              f"{verify['n_mismatch']}/{verify['n_total']} "
              f"(max gap {verify['max_logit_gap_at_mismatch']:.4f})")
        for fi, name in enumerate(names):
            ids_cut = cut_at_eos(sp, gen[fi])
            pieces = pieces_of(sp, ids_cut)
            decoded = sp.decode(ids_cut)
            w, h = sizes[fi]
            boxes = parse_boxes(pieces, w, h)
            pr["frames"][name] = {
                "decoded_text": decoded,
                "pieces": pieces,
                "n_tokens_before_eos": len(ids_cut),
                "hit_eos": len(ids_cut) < MAX_NEW,
                "boxes": boxes,
            }
            print(f"  [{name}] -> {decoded!r}  ({len(boxes)} boxes)")
            if boxes:
                safe = re.sub(r"\W+", "_", prompt)[:40]
                draw_boxes(paths_by_name[name], boxes, out_dir / f"{args.tag}_{name}_{safe}.png")
        results["prompts"][prompt] = pr

    with open(out_dir / f"results_{args.tag}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {out_dir}/results_{args.tag}.json")


if __name__ == "__main__":
    main()
