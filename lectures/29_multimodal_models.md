# 29. Multi-Modal Models: Beyond Text

## Motivation

Language is just one modality. The real world is visual, auditory, spatial, and textual simultaneously. Multi-modal models process and generate across these modalities. This lecture covers the architectural principles, the key design decisions, and the tricks that make multi-modal transformers work — with a focus on vision-language models as the dominant paradigm.

---

## 29.1 The Fundamental Problem: Tokenizing the World

### Why tokenization matters for modality unification

Transformers operate on sequences of tokens. Text already comes as a discrete sequence. The multi-modal challenge: convert continuous signals (images, audio, video) into discrete token sequences that a transformer can process.

| Modality | Raw representation | Tokenization challenge |
|---|---|---|
| Text | Character/byte sequence | Well-solved (BPE, SentencePiece) |
| Image | 2D pixel grid, 3 channels | Flatten a 2D signal into 1D sequence |
| Audio | 1D waveform at 16-48kHz | Very long sequences (1s = 16K samples) |
| Video | 3D: spatial × temporal | Astronomically long sequences |
| 3D/Point cloud | Unstructured 3D coordinates | No natural ordering |

### The universal recipe

For every modality, the pattern is the same:

1. **Split the input into patches** (spatial/temporal chunks)
2. **Project each patch to a vector** (linear projection or learned encoder)
3. **Add positional encoding** (position in the original signal)
4. **Feed into a transformer** (same attention mechanism as text)

The magic: once everything is a sequence of vectors, the transformer doesn't care where they came from. Attention over image patches works identically to attention over text tokens.

---

## 29.2 Vision Transformers (ViT): Making Images into Sequences

### Patch embedding

An image of size $H \times W \times C$ is divided into patches of size $P \times P$:

- Number of patches: $N = \frac{H \times W}{P^2}$
- Each patch is flattened: $P^2 \times C = P^2 C$ values
- Linear projection maps each flattened patch to dimension $d$: $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times d}$

For a 224×224 image with P=16: $N = 196$ patches, each of dimension $16 \times 16 \times 3 = 768$.

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel_size=stride=patch_size is equivalent to
        # extracting non-overlapping patches + linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, N)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x
```

### Positional encoding for 2D

Unlike text (1D), images have 2D structure. Options:

**1. Learned 1D positional embedding** (original ViT):
Just assign positions 0, 1, ..., N-1 to the flattened patch sequence. Surprisingly, this works well — the model learns the 2D structure from data.

**2. 2D sinusoidal encoding**:
Separate encodings for row and column, concatenated:
$$\text{PE}(i, j) = [\text{PE}_{\text{row}}(i); \text{PE}_{\text{col}}(j)]$$

Each uses the standard sinusoidal formula from Lecture 9, but applied separately to the row and column indices.

**3. RoPE-2D**:
Extend rotary position embeddings to 2D by applying separate rotations for the x and y coordinates. Used in modern vision models for resolution flexibility.

### The [CLS] token and global representation

Standard ViT prepends a learned `[CLS]` token:
$$\text{Input} = [\texttt{[CLS]}, \mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_N]$$

After the transformer, the `[CLS]` token's output serves as the global image representation (used for classification, retrieval, etc.).

**Alternative**: Global average pooling over all patch outputs (simpler, works equally well in practice).

### Why ViT needs more data than CNNs

CNNs have strong inductive biases:
- **Locality**: Convolution kernels only look at nearby pixels
- **Translation equivariance**: The same filter works everywhere

ViT has none of these — patches interact through attention from layer 1. It must **learn** locality and translation equivariance from data.

Result: ViT underperforms CNNs on small/medium datasets (ImageNet-1K alone), but **outperforms** on large datasets (ImageNet-21K, JFT-300M) where the lack of inductive bias becomes an advantage (more flexible).

---

## 29.3 Contrastive Learning: CLIP and the Alignment Revolution

### The CLIP idea (Radford et al., 2021)

Train two encoders simultaneously:
- **Image encoder** ($f_{\text{img}}$): Maps an image to a vector
- **Text encoder** ($f_{\text{txt}}$): Maps a text caption to a vector

Training objective: matching image-text pairs should have similar vectors; non-matching pairs should be dissimilar.

### Contrastive loss (InfoNCE)

Given a batch of $B$ image-text pairs $\{(I_i, T_i)\}_{i=1}^{B}$:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2B}\sum_{i=1}^{B}\left[\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B}\exp(s_{ij}/\tau)} + \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B}\exp(s_{ji}/\tau)}\right]$$

where $s_{ij} = f_{\text{img}}(I_i)^T f_{\text{txt}}(T_j)$ and $\tau$ is a learned temperature parameter.

**Interpretation**: For each image, the loss says "the correct caption should have the highest similarity among all $B$ captions in the batch." And symmetrically for each caption.

### Why CLIP works so well

1. **Massive pairing scale**: Trained on 400M image-text pairs scraped from the web. No manual labeling required — alt text, captions, etc., provide natural supervision.

2. **Batch size as negative samples**: With batch size 32K, each example has 31,999 negative pairs. This provides a very strong training signal.

3. **Zero-shot transfer**: At inference, you can classify any image by computing similarity to text templates: "a photo of a {class_name}." No fine-tuning needed on the downstream dataset.

### The temperature parameter $\tau$

$\tau$ controls the sharpness of the distribution:
- $\tau \to 0$: Only the hardest negatives matter (very peaked distribution)
- $\tau \to \infty$: All negatives contribute equally (uniform distribution)

CLIP learns $\tau$ during training (initialized to $\log(1/0.07)$). The optimal $\tau$ reflects the "difficulty" of the contrastive task.

**Interview insight**: The learned temperature in CLIP typically settles around $\tau \approx 0.01$, which is much sharper than the initial $0.07$. This means the model learns to make very confident distinctions.

---

## 29.4 Vision-Language Model Architectures

### The three design paradigms

**Paradigm 1: Dual encoder (CLIP-style)**

```
Image → Image Encoder → image_embedding
                                            → cosine similarity
Text  → Text Encoder  → text_embedding
```

- Image and text processed independently
- Interaction only via dot product at the end
- **Pro**: Very fast retrieval (pre-compute all embeddings)
- **Con**: Shallow interaction — can't reason about fine-grained image-text relationships

**Paradigm 2: Fusion encoder (Flamingo, BLIP-2)**

```
Image → Image Encoder → visual_tokens ─┐
                                        ├→ Language Model → output
Text  → Token Embed   → text_tokens   ─┘
```

- Visual tokens are injected into the language model
- Deep cross-attention between visual and textual representations
- **Pro**: Rich cross-modal reasoning
- **Con**: Can't pre-compute; need the image at inference time

**Paradigm 3: Unified decoder (GPT-4V, Gemini)**

```
Image → Visual Tokenizer → image_tokens ─┐
                                          ├→ Single Transformer → output
Text  → Text Tokenizer   → text_tokens  ─┘
```

- Everything is tokenized and fed as one sequence to a single model
- The model is just a standard autoregressive transformer operating on a mixed sequence
- **Pro**: Simplest architecture; leverages all standard transformer infrastructure
- **Con**: Longest sequences; most compute

### Which paradigm dominates?

As of 2024-2025, paradigm 3 (unified decoder) is winning for generative tasks. Paradigm 2 is used for efficiency (smaller adapter modules connecting a frozen vision encoder to a frozen LLM).

---

## 29.5 Connecting Vision to Language: The Adapter Problem

### The core challenge

A pre-trained vision encoder (e.g., ViT-L from CLIP) produces visual features. A pre-trained LLM (e.g., LLaMA) processes text. How do you connect them without destroying the capabilities of either?

### Approach 1: Linear projection (LLaVA)

The simplest approach: a single linear layer maps visual features to the LLM's input space.

$$\mathbf{h}_{\text{visual}} = \mathbf{W} \cdot f_{\text{ViT}}(\text{image}) + \mathbf{b}$$

where $\mathbf{W} \in \mathbb{R}^{d_{\text{LLM}} \times d_{\text{ViT}}}$.

**LLaVA training protocol**:
1. Stage 1: Freeze ViT + LLM, train only the projection layer on image-caption pairs (alignment)
2. Stage 2: Unfreeze LLM, fine-tune on visual instruction-following data

**Why this works**: The CLIP ViT already maps images to a space that corresponds to text concepts. The linear projection just translates this to the LLM's embedding space.

### Approach 2: Q-Former (BLIP-2)

A small transformer that "queries" the visual features using learned query vectors:

```
Learned queries (32 vectors) ──────┐
                                   ├→ Cross-Attention → 32 output vectors → LLM
ViT features (256 vectors)  ──────┘
```

The Q-Former compresses 256 ViT patch features into 32 fixed vectors, acting as an information bottleneck. This:
- Reduces sequence length (256 → 32 tokens)
- Forces extraction of the most relevant visual information
- Allows the adapter to learn what the LLM "needs to know"

### Approach 3: Perceiver Resampler (Flamingo)

Similar to Q-Former but uses a perceiver-style architecture:
- Learned latent vectors cross-attend to visual features
- Multiple layers of cross-attention refine the visual representation
- Produces a fixed number of visual tokens regardless of input resolution

### The resolution problem

Patch size determines the detail level:
- ViT with 224×224 input, P=14: 256 patches → low resolution
- ViT with 1024×1024 input, P=14: 5329 patches → high resolution but very long sequence

**High-resolution strategies**:
1. **Multi-scale encoding**: Process at multiple resolutions, concatenate features
2. **Tiled processing**: Split large images into tiles, process each separately, combine
3. **Dynamic resolution** (InternVL, LLaVA-NeXT): Adapt the number of tiles based on image aspect ratio and content

---

## 29.6 Cross-Modal Attention Patterns

### How does a VLM "look at" images while generating text?

In a fusion/unified architecture, the attention mechanism naturally creates cross-modal connections:

**What attention learns** (empirical observations):
- **Early layers**: Visual tokens attend to local patches (edges, textures)
- **Middle layers**: Cross-modal binding — text tokens for "dog" attend to the dog region in the image
- **Late layers**: Reasoning — the model integrates visual evidence with textual context

### Gated cross-attention (Flamingo)

Rather than letting all layers attend to visual features, Flamingo adds gated cross-attention at specific layers:

$$\text{Output} = \text{SelfAttn}(x) + \tanh(\alpha) \cdot \text{CrossAttn}(x, v)$$

where $\alpha$ is initialized to zero, so the model starts as a pure language model and gradually learns to incorporate visual information. This provides stable training and preserves the LLM's language capabilities.

### Attention sink in multi-modal models

The first visual token often becomes an "attention sink" — receiving disproportionate attention when the information isn't needed. This is analogous to the `[BOS]` attention sink in text-only models (see Lecture 23).

---

## 29.7 Vision Generation: The Diffusion Connection

### Visual tokenizers for generation

For models that **generate** images (not just understand them), you need a visual tokenizer that maps images to discrete tokens and back.

**VQ-VAE (Van den Oord et al., 2017)**:
1. Encoder maps image to continuous latent codes
2. Codes are quantized to nearest entries in a learned codebook
3. Decoder reconstructs the image from codebook entries

$$\text{Image} \xrightarrow{\text{Encoder}} \mathbf{z}_e \xrightarrow{\text{Quantize}} \mathbf{z}_q \xrightarrow{\text{Decoder}} \hat{\text{Image}}$$

Training loss: $\mathcal{L} = \|\text{Image} - \hat{\text{Image}}\|^2 + \|\text{sg}[\mathbf{z}_e] - \mathbf{z}_q\|^2 + \beta\|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|^2$

where $\text{sg}[\cdot]$ is stop-gradient. This trains the encoder, codebook, and decoder jointly.

### Autoregressive image generation

Once images are discrete tokens, you can use a standard autoregressive transformer:

$$P(\text{image}) = \prod_{i=1}^{N} P(t_i | t_1, \ldots, t_{i-1})$$

**The ordering problem**: Pixels/patches have 2D structure but autoregressive models need a 1D ordering. Common choices:
- Raster scan (left-to-right, top-to-bottom)
- Hilbert curve (locality-preserving space-filling curve)
- Random ordering (works with masked models)

### Diffusion models vs autoregressive for vision

| Property | Autoregressive | Diffusion |
|---|---|---|
| Quality | Good | Excellent |
| Speed | Slow ($N$ sequential steps) | Medium (~50 denoising steps) |
| Multi-modal integration | Natural (same architecture) | Requires separate text encoder |
| Controlability | Token-level | Noise-level, conditioning |
| Current dominance | Text generation | Image generation |

Emerging trend: **unified models** (Chameleon, Transfusion) that do autoregressive text + diffusion-based image in a single architecture.

---

## 29.8 Audio and Speech

### Audio tokenization

Audio presents unique challenges:
- Very long sequences (16kHz × 30s = 480K samples)
- Both content (what is said) and style (how it's said) matter

**Approach 1: Spectrogram patches** (Whisper):
- Convert audio to mel-spectrogram (time × frequency bins)
- Process as a 2D image with patch embeddings
- Whisper uses 80 mel bins × 3000 frames (30s at 10ms hop), patches of size 2×80

**Approach 2: Learned discrete tokens** (EnCodec, SoundStorm):
- Neural audio codec compresses audio to discrete codes at ~1.5kbps
- Multiple "codebook levels" capture coarse-to-fine detail
- Level 1: content/phonemes; Levels 2-8: acoustic details, speaker identity

**Approach 3: Continuous features** (HuBERT, wav2vec2):
- Self-supervised speech encoder produces continuous frames
- Optionally discretized via k-means clustering

### Speech-language models

Pattern: Encoder produces audio tokens → interleave with text tokens → single transformer processes both.

Example (GPT-4o style):
```
[audio_tokens_1...audio_tokens_T, <text_start>, text_tokens..., <audio_start>, audio_output_tokens...]
```

The model sees audio and text as parts of one sequence and can generate in either modality.

---

## 29.9 Practical Considerations

### Compute budget across modalities

For a VLM processing an image (256 visual tokens) with a text prompt (100 tokens):
- Prefill: 356 tokens total → quadratic attention cost is tolerable
- The visual tokens are ~2.5× the text tokens but carry proportionally more information

**Cost model**: Visual encoding (ViT forward pass) is a one-time cost per image. The expensive part is the cross-modal attention in the LLM, which scales with $O(N_{\text{visual}} \times N_{\text{text}})$.

### Resolution-quality tradeoff

More pixels → more patches → better detail but higher cost. In practice:

| Resolution | Patches (P=14) | Detail level | Use case |
|---|---|---|---|
| 224×224 | 256 | Low | Classification, retrieval |
| 448×448 | 1024 | Medium | General VQA |
| 768×768 | 3025 | High | Document understanding, OCR |
| 1024+ | 5000+ | Very high | Dense visual reasoning |

### Hallucination in VLMs

VLMs hallucinate visual content more than text-only models hallucinate facts. Common failure modes:
- **Object hallucination**: "There is a clock on the wall" when there isn't one
- **Attribute binding errors**: Correct objects, wrong attributes ("the blue car" when the car is red)
- **Spatial relationship errors**: "The cat is on the table" when the cat is under the table

**Why**: The visual features compress spatial information, and fine-grained spatial reasoning is hard. The LLM's language prior is strong and can override weak visual evidence — the model "says what it expects to see" rather than what it actually sees.

### Evaluation for multi-modal models

| Benchmark | Tests | Key metric |
|---|---|---|
| VQAv2 | Visual question answering | Accuracy |
| TextVQA | OCR + reasoning | Accuracy |
| POPE | Object hallucination | F1 score |
| MMBench | Multi-modal understanding | Aggregated score |
| MMMU | Expert-level multi-modal | Accuracy |
| MathVista | Visual math reasoning | Accuracy |

---

## 29.10 The Multi-Modal Scaling Hypothesis

### Why multi-modal might be necessary for AGI

The argument: language alone is an impoverished representation of the world. A model trained only on text has no grounding — it knows that "red" relates to "tomato" but has never seen either.

Multi-modal training provides:
1. **Grounding**: Connecting symbols to sensory experience
2. **Cross-modal transfer**: Visual reasoning improves language reasoning and vice versa
3. **Richer world models**: 3D spatial understanding, physics intuitions, temporal reasoning

### The Platonic Representation Hypothesis

Huh et al. (2024): As models get larger and are trained on more data across modalities, their internal representations **converge** to a shared statistical structure that reflects the underlying reality.

Evidence: CLIP's image features and GPT's text features, when aligned, produce similar geometric structures. Different modalities trained on different data converge to similar representations.

**Implication**: There may be a universal representation of the world that all sufficiently powerful models discover, regardless of the modality they're trained on. Multi-modal training accelerates convergence to this representation.

### Current frontier: "Omni" models

GPT-4o, Gemini, etc. process text, images, audio, and video in a single model. The architectural trend: a single transformer with modality-specific tokenizers but a shared "reasoning core."

The prediction: future models will add more modalities (touch, 3D, robotics actions) with the same pattern — tokenize, project, attend.
