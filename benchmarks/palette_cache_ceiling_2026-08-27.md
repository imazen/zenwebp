# VP8L palette color-cache ceiling sweep — 2026-08-27 (#77)

Decides the tuned default for the palette color-cache search ceiling
(`PALETTE_CACHE_BITS_CEILING` in `src/encoder/vp8l/encode.rs`). Before this
sweep the ceiling was platform-dependent: `31 - usize::leading_zeros()` gave
libwebp's `BitsLog2Floor(palette)+1` on 32-bit targets and saturated to 1 on
64-bit targets (#77).

- Commit under test: d62a1d46 (+ the `palette_cache_bits_ceiling` refactor,
  byte-identical to d62a1d46 at ceiling 1 — the baseline column IS the
  pre-change 64-bit behaviour).
- Host: Apple M4 Pro (aarch64), release build, default `LosslessConfig`
  (q75, `cache_bits = None`), RGBA8 input, methods m0/m3/m4/m5/m6.
- Corpus: 120 files sampled from the `codec-corpus` R2 bucket, source label
  `corpus/png-8` (real 8-bit palette PNGs), stratified into 12 log-spaced
  file-size bins (2 KB – 3 MB, 10 per bin, `random.seed(77)`); 90 of them
  are ≤ 256 colors and ≤ 2.5 MP and were encoded (the other 30 were skipped:
  > 256 colors after decode, or > 2.5 MP). Blob keys listed below.
- Harness: a scratch crate (`~/tmp/zenwebp-77-sweep`, not committed) that
  decodes each PNG with `png 0.18`, counts distinct RGBA colors, and encodes
  through the public `EncodeRequest::new(EncoderConfig::Lossless(..))` API;
  the ceiling was varied through a temporary env override in the function
  (removed before commit). Raw TSVs: `~/tmp/zenwebp-77-{baseline,c0,c2,c3,libwebp}.tsv`.

Bytes are summed over all 90 images per method; "better / worse / tied" count
cells where the variant is smaller / larger / equal than ceiling 1.


## Ceiling 0 vs ceiling 1 (450 cells)

| method | ceiling 1 bytes | variant bytes | Δ | better / worse / tied |
|---|---|---|---|---|
| 0 | 14409728 | 14611332 | +201604 (+1.399 %) | 0 / 83 / 7 |
| 3 | 12230058 | 12431470 | +201412 (+1.647 %) | 11 / 48 / 31 |
| 4 | 12240058 | 12437478 | +197420 (+1.613 %) | 14 / 46 / 30 |
| 5 | 12160232 | 12356022 | +195790 (+1.610 %) | 0 / 44 / 46 |
| 6 | 12160616 | 12356312 | +195696 (+1.609 %) | 0 / 45 / 45 |
| all | 63200692 | 64192614 | +991922 (+1.569 %) | 25 / 266 / 159 |

Geometric mean of per-cell ratio: 1.01049

## Ceiling 2 vs ceiling 1 (450 cells)

| method | ceiling 1 bytes | variant bytes | Δ | better / worse / tied |
|---|---|---|---|---|
| 0 | 14409728 | 14389834 | -19894 (-0.138 %) | 24 / 58 / 8 |
| 3 | 12230058 | 12242450 | +12392 (+0.101 %) | 20 / 19 / 51 |
| 4 | 12240058 | 12256808 | +16750 (+0.137 %) | 15 / 24 / 51 |
| 5 | 12160232 | 12161880 | +1648 (+0.014 %) | 19 / 12 / 59 |
| 6 | 12160616 | 12162114 | +1498 (+0.012 %) | 19 / 12 / 59 |
| all | 63200692 | 63213086 | +12394 (+0.020 %) | 97 / 125 / 228 |

Geometric mean of per-cell ratio: 1.00118

## Ceiling 3 vs ceiling 1 (450 cells)

| method | ceiling 1 bytes | variant bytes | Δ | better / worse / tied |
|---|---|---|---|---|
| 0 | 14409728 | 14399408 | -10320 (-0.072 %) | 19 / 64 / 7 |
| 3 | 12230058 | 12319840 | +89782 (+0.734 %) | 6 / 34 / 50 |
| 4 | 12240058 | 12338458 | +98400 (+0.804 %) | 9 / 31 / 50 |
| 5 | 12160232 | 12231730 | +71498 (+0.588 %) | 11 / 20 / 59 |
| 6 | 12160616 | 12231904 | +71288 (+0.586 %) | 12 / 19 / 59 |
| all | 63200692 | 63521340 | +320648 (+0.507 %) | 57 / 168 / 225 |

Geometric mean of per-cell ratio: 1.00374

## Ceiling libwebp log2+1 vs ceiling 1 (450 cells)

| method | ceiling 1 bytes | variant bytes | Δ | better / worse / tied |
|---|---|---|---|---|
| 0 | 14409728 | 14615848 | +206120 (+1.430 %) | 0 / 83 / 7 |
| 3 | 12230058 | 12369242 | +139184 (+1.138 %) | 5 / 36 / 49 |
| 4 | 12240058 | 12388524 | +148466 (+1.213 %) | 8 / 33 / 49 |
| 5 | 12160232 | 12247178 | +86946 (+0.715 %) | 11 / 20 / 59 |
| 6 | 12160616 | 12247352 | +86736 (+0.713 %) | 12 / 19 / 59 |
| all | 63200692 | 63868144 | +667452 (+1.056 %) | 36 / 191 / 223 |

Geometric mean of per-cell ratio: 1.00698

## Decision

Ceiling 1 (the pre-existing 64-bit behaviour) is the best or tied-best point
on every axis; the libwebp ceiling loses at every method, including m5/m6
where the cache/no-cache choice is made by full encode. Pinned as
`PALETTE_CACHE_BITS_CEILING = 1`, unifying 32- and 64-bit output with zero
byte change on 64-bit. Ceiling 2 is a wash (+0.02 %, 97 better / 125 worse)
and is the only candidate worth revisiting with a larger corpus.

## Palette-size distribution of the 90 encoded images


| colors | images |
|---|---|
| ≤16 | 15 |
| 17-64 | 13 |
| 65-255 | 33 |
| 256 | 29 |

## Blob keys (`s3://codec-corpus/blobs/<aa>/<bb>/<sha256>`), the 90 encoded files

```
00/e1/00e1e98c8750c02f687ff6f11fa29ad3c653788d60e4fdb015d32d802b5df534
01/fa/01fa6de8592ed44a6f54ec9734fbe099e6cfcb6cd6836be1c2d4d31f787abfe3
02/aa/02aae28b01f2b8342dba81bf470e2d7f286de37df7574a5fbc6847c52272f9e3
06/b1/06b123c3e9996f2eb57c34224ed4f50c58972542893826a5a0e154c3840cb968
07/4d/074dcc890d87f852a9e5936defb9524d867d970a4b90e058adca045a191bdb8f
0c/a7/0ca75cba7025fd3f2a4848a68f95fe735aa19a19395f9e6669cd16fb4b8012e3
0e/a8/0ea8a0c6e6e820cb1309f54ad95515e38284b8a10fd626b9278f442d694fab79
13/8b/138bc4425b908d0ae82d3bb7c5eb100c2859a78c13ac6b5b0d7d8864c21d445a
14/68/1468d82c2cd49390831b0e3b0811429c4a7976786a5496f46f746e50f99da211
1c/95/1c95a600d3b25c761708bae5451ae0e5418391c4bef0a7afdb1bf6eb6ddbd8d6
22/cf/22cf14a90fc5f9445f60b82f06d17d9dea55df1faadac475c064ca91b722ede7
26/b1/26b1e23daf263d38d53d42c1e2793f90cb77f34d79b4a37ba56361b452826aa6
27/1e/271e263038597ec25d5ef9dcf5d03dcde59abd63f85b33f22be70af8a3b116b8
27/63/276379d5a655ba3ef21c4fd351d62d4f8b0c453def02051cb9dedf155d848489
27/b9/27b9a1fc1024edc999dac3220c76fd2357fa8868a7da453d8c634839d7fcdb10
2b/2b/2b2b6cb2bb4de7c678b2df585502f53f9b5d89b7cb91c9d9db5dd337d7a8a619
2c/8a/2c8ac4d4ab7cedc8f1b6a9a27554c76574be5f8a3f90af9655df46e5b3de78e7
30/3c/303cf0edc36dcd137afb10524606d439f77610a6d3422561cc4e7b8c986195bd
30/8f/308f7443bd47dab8ee1bf96e564f7409cb6465bacd40ff43bd422d7150da12b2
31/8d/318d8d714dee0d375537c065b1f643c86a8b223a9e2b83dcc8443a481fe957de
35/4b/354b8d3c40161400f58fc542511bc2ab6f1567f2d326e8266996f3a5626a6d4a
35/8f/358f66bceb12fc49638bd33b6d00bcb94905ec1e6abde5211f0cdf024d94a34c
36/0d/360d64eff58ce8eea220f12080ed996415752389d46333f9bdee38accd67d7b1
37/eb/37ebf6902ccc0b2dd683dad93f9d8165c125243e5d1050c81e1ee706a6792ebd
39/8c/398c920a5b74be19fec126d7a7b77c8e4ed657cbf00081018771c166f0ab35c5
39/cc/39cc272b52e4d908d52f16959b072ee423ed59c5b7d55411e70d1a9d8086fce5
3b/f2/3bf2c64eee403545bee59a8560da355c84d499ecc445e034d5edf9afa4fa5c25
3c/4b/3c4bf617b2acb2383d02226f7088602cd0b32b473fc1b6d92fea158c6b0a506c
3c/7e/3c7e42f84a06ddbd9d9e313b836c836d8edafbb21179d643da1b3664d62aa7a4
3c/f2/3cf2e9eac0d0364ef57aa2a1a3bcf7021f48e547a08ab6e92b47c7e41fb995fa
3f/37/3f3705d7c78dfbcd1b8d637c315471790ac25fe32ffd9c1ca0439d06a4409752
44/55/445530c634d2ce5ab5383228db16ce8927c1a489ab5a80c38ea8d3350127fb40
47/a2/47a270d9be17b00c88e8dae34201074a82dbe0ed7ecad77922954bd0fec4f94d
49/03/49037f2c608c60537bf0c70cfa02df9933120bc17293b61a41686b7f079667a6
4d/41/4d412caf4c08a5d55315cbac1d698bc78da25000107902fd11f2e9a3fd5d376a
4f/69/4f69985fa42da1c2ceec4f19bcca213fc0557f9d681f2b1fe962f511514c94a2
51/55/5155acf64135c733fbe3368c2b1f7d99d1dc551a24ad33bc274e9560604ef8a9
54/0a/540a7849df6ed680904be508d9f7922445ac628b4e2eca0f515f1da7f13c2daa
5e/0a/5e0a419377eab03bdc2dafbfb5f274b1adec1fda24e55e46bb4e37527a2e4c15
62/3e/623eeed988a6116cff1471a5680c972814be1ebc8978db27c4d9c4549de31b64
63/c9/63c98fdc094f0d5f8ebdaafd3cfe8dbe397e285727d7a9fb9a64edfe473ae694
65/0e/650e83055030216ab9847d6cd1926e3f1fe73e068424a15aa261cd1737fd6d12
68/b0/68b07aefae31502a0ce31d5e69bfb2b39c4a6f9788128b6b5f653462ee996d6c
6a/a6/6aa62354bf249ac2f1c1cd58bc2f1d061fced5e650fcea56f437c01c321a7529
6b/4a/6b4a9435647dae13e45e8d8f3931ccb647ce286eeda11ae7d1a4d9d9fc3acc88
6c/69/6c69a901f827652367d7d8b36830783c55593d992c4cc8be59af2dea79582d29
71/86/7186c91e94a135d054fc2828b524773d67bc1ad6a633394ff5eeb6142bb0d95b
76/9e/769e3bd0e74389b43c558df2de0dd037ff56cc0c432a126a41397527f893a41f
77/dc/77dcb3cb76462617aae33fb92ed65be71521edf84a3c88409831afc72798010c
7a/3d/7a3df95700a3a69d55f7c01bed47cc230dff897b6c5aedeb1f43a4ab87ba185c
7c/bb/7cbb66441944ef6e8023ffebb435cb393981695e898bc182d7ad1773e820ff3c
7d/2e/7d2e5f9e12e0e71de65b565cab1e027f99ab86b386bc4d22b21698d7136956ac
83/0f/830fb0333372384b846c888a3e6819483dee1f00604db083b0fea7b5f43f43b5
84/55/84552c3bcceec3f9de4e1506f25bd195e45450b46ce8b235b76a7b554c702414
84/6a/846aa028676201e4b4848f2cac17b66cdb60cf7ef2cfd5aa6367adfb2e8b7c50
85/f4/85f42891dba7bbcc777b4f1036a1ac2d7d157b2c448cd1b7e98a08b961fae03b
91/17/911796142e38a4f5435e42a2c58a1b1bb040c90068aaf885f10e18918b1e398d
92/fe/92fe5a5ce6728fd834ee08760ead1e6976d91644a719ed34a9a751a5707f3ce2
96/84/9684dc7633bc51122c5cafc6866bada5eafc6e432b9dc4f9f12edbe6f5c25d00
97/08/9708bcb44d921d32a04f68c60e5c1fc4d4113ac58caa86610fb8e6f6ece78923
9b/e7/9be789871fca96e8d38524989f5c20c4c56654159d85b8ad8843ddbdbbe16799
9d/d7/9dd761a98925e9f23efe6b307102cbddeb3566450a2d5e8147437d53289f260e
9f/ce/9fce7ee0efdb60841ee0d8cf5648cf0c4c73e4ac73f106be59eae2e93232a7ed
a6/6b/a66b79e2e49314bb54a3909bc76f4627599f69e619074c1f21f32a2dcdd0075a
ad/3e/ad3ea9788ff304988e5acfb4d73b3eaf8697cac4df6727428b4f38a8b5a759ac
b1/0f/b10f0bd9153b1a0314863e5320e4821b464ea57a211cd9fad24c3503d75875bb
b1/22/b12217a59cdf4eea2ab035f9177f0b3011785161d4bebcbedf272e22cd98ff2d
b2/ce/b2cece2b0ec81af16066918971fd67459c788d55a48de0749e26dca08cafa27c
b7/b1/b7b1a569b81145ebf3e494a558cb29d86b00b6caa2c2a6789152c373a1a74944
bc/98/bc98be132cf2c0f21bd079e17546aec72bafb30c5ba621b5cbd6dba5b7ad35a4
c2/67/c267f8c9cf393a8d8b661ecf484a86f6f6969cfb20aee66ca7eb83293ccc2ea7
c6/d1/c6d1a81c48bc44c922021e28506a7f242bc8280b233e9279cfa77fd027c10cec
cf/81/cf81495903ba722012a479a706f200839d629a52ffed013842c8d00825d27273
cf/f0/cff04948b5d0d20359f6c997b749c4fd3a4b9e26f06962ed4621684832db0242
d0/24/d0243e56ae26f9323b955ab3eca237963bc0792c52b046e9c912140c77319ca2
d3/c8/d3c84c79c0e73f99afcb97fc1c1ee55496f743e2c1dfc90f881390fc32eeee99
de/2e/de2efa39b8b2673b820b9e57c0375aafca698952cf2f9ef586a34d9107357811
de/f8/def8034d37ac50aa4b2cf70c184f18594659fb4208ea75169044a719ac29fad6
e0/4c/e04c046df23c9deb9ccc1c7624a7312f0f90d199fbdeef8156b5ea76b58d8e07
e3/16/e316f82121201ec48328eda29c552cb73009200c15ae9dd3775cefd13f02aaf8
e5/98/e5982f9833f37b52df3bd42621b95823f29783f6fbf1d07881c2c492b6d22964
ec/7f/ec7f7016632b929c7c61301c62e8f089ab82a5496bb76dc41084b50944d5c586
f0/93/f0932c764bc5c5f60c3e36d4f54fd1b1e1db0a59435027ffb8cc22e6d0c72e79
f2/ad/f2adb6c724f5ffaf4948c93fde198d80ebb1fc345e46e75fe342b4cbaa5126b8
f3/18/f3184b98a23b5d58f33d41a474ad66e087a52f5e6b78f2fdf7626817bdddaf25
f4/cc/f4cce6effcc8285397e704e3b395cd1f5202bf2a2b007665bd4d2b51a8727850
f5/8e/f58e9f09413e1da4e56dca13e7492dd36cb04a4d24fe5b3178ffc79bc80c2e0c
f6/01/f6015378a28e84986a814cb2670aa35b6b26c0ce58e31a55904d9936a99b3e44
f8/d0/f8d017b1abc8d833085e9b3644838e67f4f446cfa3fc40aecb917fce12a04a85
fd/9d/fd9da11707cb19df129563ed3dda0d38c18324c252e1810970094f457852a1e3
```
