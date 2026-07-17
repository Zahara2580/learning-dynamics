embedding shape: (384, 1536)



=== per-range input-embedding diff (L2 per row, nguni vs byt5) ===

range                                                    mean        max  #exact-zero

specials 0-2 (pad/eos/unk)                             0.2403     0.3968            1

bytes 3-133 (ASCII incl. letters)                      0.2350     0.5335           32

bytes 134-258 (high bytes = paper sentinel zone)       0.3124     0.7886           18

extras 259-320 (HF extra_id low end)                   1.3495     3.3215            0

extras 321-383 (HF extra_id high end)                  1.5340     3.0551            0



(reference: trained ASCII byte rows average 0.2350 - rows well below this were touched rarely or never)



=== row-by-row around the 258/259 boundary ===

  row 250: emb diff = 0.00000   head diff = 3.16351

  row 251: emb diff = 0.00000   head diff = 4.00136

  row 252: emb diff = 0.00000   head diff = 4.31454

  row 253: emb diff = 0.00000   head diff = 4.33553

  row 254: emb diff = 0.00000   head diff = 3.87204

  row 255: emb diff = 0.00000   head diff = 4.35440

  row 256: emb diff = 0.00000   head diff = 3.88008

  row 257: emb diff = 0.00000   head diff = 4.17022

  row 258: emb diff = 0.00000   head diff = 1.45412

  row 259: emb diff = 0.39366   head diff = 2.77012

  row 260: emb diff = 0.55116   head diff = 3.27228

  row 261: emb diff = 0.50360   head diff = 3.43508

  row 262: emb diff = 0.51277   head diff = 3.32784

  row 263: emb diff = 0.54689   head diff = 3.24665

  row 264: emb diff = 0.55773   head diff = 3.21139

  row 265: emb diff = 0.53679   head diff = 3.35231

  row 266: emb diff = 0.56631   head diff = 3.32065

  row 267: emb diff = 0.53376   head diff = 3.44006

  row 268: emb diff = 0.67257   head diff = 3.36107



=== row-by-row at the top of the vocab ===

  row 370: emb diff = 1.43155   head diff = 2.84478

  row 371: emb diff = 1.35814   head diff = 2.90260

  row 372: emb diff = 1.17542   head diff = 2.81938

  row 373: emb diff = 0.96095   head diff = 2.81726

  row 374: emb diff = 0.89973   head diff = 2.63879

  row 375: emb diff = 1.16650   head diff = 2.63054

  row 376: emb diff = 1.23227   head diff = 2.64347

  row 377: emb diff = 0.91268   head diff = 2.44621

  row 378: emb diff = 0.83204   head diff = 2.52106

  row 379: emb diff = 0.35896   head diff = 2.44818

  row 380: emb diff = 0.10812   head diff = 2.44876

  row 381: emb diff = 0.03305   head diff = 2.40855

  row 382: emb diff = 0.01097   head diff = 2.50465

  row 383: emb diff = 0.00885   head diff = 2.51408



=== top 25 most-changed rows in 134-383 (sentinel candidate zone) ===

  row 318: emb diff = 3.32146

  row 320: emb diff = 3.20179

  row 321: emb diff = 3.05505

  row 319: emb diff = 2.87932

  row 313: emb diff = 2.66755

  row 312: emb diff = 2.60217

  row 311: emb diff = 2.50871

  row 316: emb diff = 2.50167

  row 305: emb diff = 2.45569

  row 314: emb diff = 2.39839

  row 342: emb diff = 2.38622

  row 349: emb diff = 2.36818

  row 322: emb diff = 2.34821

  row 323: emb diff = 2.34812

  row 306: emb diff = 2.34285

  row 317: emb diff = 2.31644

  row 341: emb diff = 2.30476

  row 340: emb diff = 2.27619

  row 335: emb diff = 2.24010

  row 347: emb diff = 2.21239

  row 333: emb diff = 2.18661

  row 348: emb diff = 2.18506

  row 324: emb diff = 2.15822

  row 334: emb diff = 2.13520

  row 343: emb diff = 2.10223



=== VERDICT SIGNALS ===

extras low end  (259-289) mean diff: 0.8059

extras high end (353-383) mean diff: 1.0821

high-byte zone  (200-258) mean diff: 0.2752

trained-byte baseline (ASCII)      : 0.2350



PATTERN: mixed/unclear - read the row-by-row tables above manually.

rmdrak003@srvrochpc100 /scratch/rmdrak003/l

rmdrak003@srvrochpc100 /scratch/rmdrak003/learning-dynamics$ uv run python3 scripts/memory_fit_checks/diagnose_nguni_lafand_style.py



=== francois-meyer/nguni-byt5-large (data: /scratch/rmdrak003/data/preprocessed/nguni-byt5) ===

chunk length 568, batch 4

mean masked-run length across batch: 1.149 (lafand geometric expectation ~1.18); runs per example ~74

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

[transformers] `torch_dtype` is deprecated! Use `dtype` instead!

Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 73952.75it/s]

[transformers] The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.

A: descending from 383 (T5-style / our base-384)

  -> loss = 5.0660

B: ascending from 259  (<extra_id_0>=259)

  -> loss = 1.5667

C: descending from 258 (lafand as written)

  -> loss = 8.7868

  //actuall loss doesnt make sense , more important is the curve. sentinel token is a sign and input and generate the words being masked, if you use the same sentinal token that byt5 , 

  //use the afribyt5 code , nguni and byt5, afrimt5 lafan repo. use this code for everything. lean more towards research code because they have done everything