//! Exact Rust port of libwebp's SharpYUV converter (`sharpyuv/*.c`), 8-bit
//! path only — the operating point `WebPPictureSharpARGBToYUVA` uses
//! (`rgb_bit_depth=8`, `yuv_bit_depth=8`, `kSharpYuvMatrixWebp`, sRGB
//! transfer). Byte-identical output to libwebp's `SharpYuvConvert` — this is
//! what `use_sharp_yuv=1` produces, and `StrictLibwebpParity` routes here
//! (#38).
//!
//! Algorithm (Skal's "sharp RGB→YUV"): work in a 10-bit fixed-point
//! W/(R−W,G−W,B−W) representation (`GetPrecisionShift(8)` = 2 extra bits).
//! The *target* luma per pixel and target chroma per 2×2 block are computed
//! in LINEAR light (sRGB↔linear via 16-bit fixed-point tables), then up to
//! [`K_NUM_ITERATIONS`] gradient steps adjust the full-resolution luma plane
//! and half-resolution chroma deltas so that the DECODER's 9-3-3-1 fancy
//! upsampling filter reconstructs pixels whose linear-light luma matches the
//! target. Iteration stops early when the summed |luma error| falls below
//! 3·w·h or grows. The final W/RGB planes convert to YUV through the WebP
//! matrix in 16.16 fixed point.
//!
//! The scalar kernels here are bit-equivalent to libwebp's SSE2 versions
//! (`sharpyuv_sse2.c`): `UpdateY`/`UpdateRGB` are exact integer ops, and
//! `FilterRow16_SSE2`'s two-stage shift `(a0 + ((a0+3a1+3b0+b1+8)>>3))>>1`
//! equals the C `(9a0+3a1+3b0+b1+8)>>4` by the nested floor-division
//! identity, so matching the C reference matches the SIMD build too.

use alloc::vec;
use alloc::vec::Vec;

/// libwebp `kNumIterations`.
const K_NUM_ITERATIONS: usize = 4;

/// libwebp `YUV_FIX`: fixed-point precision for RGB→YUV.
const YUV_FIX: u32 = 16;
const K_YUV_HALF: i64 = 1 << (YUV_FIX - 1);

/// `GetPrecisionShift(8)`: two extra precision bits (8+2 ≤ kMaxBitDepth 14).
const SFIX: u32 = 2;
/// Working bit depth: `rgb_bit_depth + GetPrecisionShift(rgb_bit_depth)`.
const Y_BIT_DEPTH: u32 = 8 + SFIX;
/// `(1 << Y_BIT_DEPTH) - 1`.
const MAX_Y: i32 = (1 << Y_BIT_DEPTH) - 1;

// `kSharpYuvMatrixWebp` (sharpyuv_csp.c) with the offset column pre-shifted
// by SFIX exactly as `SharpYuvConvertWithOptions` does when
// rgb_bit_depth == yuv_bit_depth (rows copied, offsets `Shift(v, sfix)`).
const RGB_TO_Y: [i32; 4] = [16839, 33059, 6420, (16 << 16) << SFIX];
const RGB_TO_U: [i32; 4] = [-9719, -19081, 28800, (128 << 16) << SFIX];
const RGB_TO_V: [i32; 4] = [28800, -24116, -4684, (128 << 16) << SFIX];

pub(crate) const GAMMA_TO_LINEAR_TAB: [u32; 1026] = [
    0, 14, 28, 43, 57, 71, 85, 100, 114, 128, 142, 156, 171, 185, 199, 213, 228, 242, 256, 270,
    284, 299, 313, 327, 341, 356, 370, 384, 398, 412, 427, 441, 455, 469, 484, 498, 512, 526, 540,
    555, 569, 583, 597, 612, 626, 640, 654, 668, 683, 697, 711, 725, 740, 754, 768, 782, 796, 811,
    825, 839, 853, 868, 882, 896, 910, 924, 939, 953, 967, 981, 996, 1010, 1024, 1038, 1052, 1067,
    1081, 1095, 1109, 1124, 1138, 1152, 1166, 1180, 1195, 1209, 1223, 1238, 1253, 1267, 1282, 1297,
    1312, 1327, 1342, 1358, 1373, 1389, 1404, 1420, 1436, 1451, 1467, 1483, 1500, 1516, 1532, 1549,
    1565, 1582, 1599, 1615, 1632, 1649, 1666, 1684, 1701, 1718, 1736, 1754, 1771, 1789, 1807, 1825,
    1843, 1861, 1880, 1898, 1916, 1935, 1954, 1973, 1991, 2010, 2029, 2049, 2068, 2087, 2107, 2126,
    2146, 2166, 2186, 2206, 2226, 2246, 2266, 2287, 2307, 2328, 2348, 2369, 2390, 2411, 2432, 2453,
    2475, 2496, 2518, 2539, 2561, 2583, 2605, 2627, 2649, 2671, 2693, 2716, 2738, 2761, 2783, 2806,
    2829, 2852, 2875, 2899, 2922, 2945, 2969, 2992, 3016, 3040, 3064, 3088, 3112, 3136, 3161, 3185,
    3210, 3235, 3259, 3284, 3309, 3334, 3359, 3385, 3410, 3436, 3461, 3487, 3513, 3539, 3565, 3591,
    3617, 3644, 3670, 3697, 3723, 3750, 3777, 3804, 3831, 3858, 3885, 3913, 3940, 3968, 3996, 4024,
    4051, 4079, 4108, 4136, 4164, 4193, 4221, 4250, 4279, 4308, 4337, 4366, 4395, 4424, 4454, 4483,
    4513, 4543, 4573, 4603, 4633, 4663, 4693, 4723, 4754, 4785, 4815, 4846, 4877, 4908, 4939, 4971,
    5002, 5033, 5065, 5097, 5128, 5160, 5192, 5225, 5257, 5289, 5322, 5354, 5387, 5420, 5453, 5486,
    5519, 5552, 5585, 5619, 5652, 5686, 5720, 5754, 5788, 5822, 5856, 5890, 5925, 5959, 5994, 6029,
    6064, 6099, 6134, 6169, 6204, 6240, 6275, 6311, 6347, 6383, 6418, 6455, 6491, 6527, 6564, 6600,
    6637, 6674, 6710, 6747, 6785, 6822, 6859, 6897, 6934, 6972, 7010, 7047, 7085, 7124, 7162, 7200,
    7239, 7277, 7316, 7355, 7394, 7433, 7472, 7511, 7550, 7590, 7629, 7669, 7709, 7749, 7789, 7829,
    7869, 7910, 7950, 7991, 8031, 8072, 8113, 8154, 8195, 8237, 8278, 8320, 8361, 8403, 8445, 8487,
    8529, 8571, 8614, 8656, 8699, 8741, 8784, 8827, 8870, 8913, 8956, 9000, 9043, 9087, 9130, 9174,
    9218, 9262, 9306, 9351, 9395, 9439, 9484, 9529, 9574, 9619, 9664, 9709, 9754, 9800, 9845, 9891,
    9937, 9983, 10029, 10075, 10121, 10167, 10214, 10260, 10307, 10354, 10401, 10448, 10495, 10543,
    10590, 10637, 10685, 10733, 10781, 10829, 10877, 10925, 10974, 11022, 11071, 11119, 11168,
    11217, 11266, 11315, 11365, 11414, 11464, 11513, 11563, 11613, 11663, 11713, 11763, 11814,
    11864, 11915, 11965, 12016, 12067, 12118, 12169, 12221, 12272, 12324, 12375, 12427, 12479,
    12531, 12583, 12635, 12688, 12740, 12793, 12846, 12899, 12952, 13005, 13058, 13111, 13165,
    13218, 13272, 13326, 13380, 13434, 13488, 13542, 13597, 13651, 13706, 13761, 13815, 13870,
    13926, 13981, 14036, 14092, 14147, 14203, 14259, 14315, 14371, 14427, 14484, 14540, 14597,
    14653, 14710, 14767, 14824, 14881, 14939, 14996, 15054, 15111, 15169, 15227, 15285, 15343,
    15401, 15460, 15518, 15577, 15636, 15695, 15754, 15813, 15872, 15931, 15991, 16050, 16110,
    16170, 16230, 16290, 16350, 16411, 16471, 16532, 16593, 16653, 16714, 16775, 16837, 16898,
    16959, 17021, 17083, 17144, 17206, 17268, 17331, 17393, 17455, 17518, 17581, 17643, 17706,
    17769, 17833, 17896, 17959, 18023, 18087, 18150, 18214, 18278, 18342, 18407, 18471, 18536,
    18600, 18665, 18730, 18795, 18860, 18925, 18991, 19056, 19122, 19188, 19254, 19320, 19386,
    19452, 19519, 19585, 19652, 19718, 19785, 19852, 19920, 19987, 20054, 20122, 20189, 20257,
    20325, 20393, 20461, 20529, 20598, 20666, 20735, 20804, 20873, 20942, 21011, 21080, 21149,
    21219, 21289, 21358, 21428, 21498, 21568, 21639, 21709, 21780, 21850, 21921, 21992, 22063,
    22134, 22205, 22277, 22348, 22420, 22492, 22564, 22636, 22708, 22780, 22853, 22925, 22998,
    23071, 23143, 23217, 23290, 23363, 23436, 23510, 23584, 23657, 23731, 23805, 23880, 23954,
    24028, 24103, 24178, 24253, 24327, 24403, 24478, 24553, 24629, 24704, 24780, 24856, 24932,
    25008, 25084, 25160, 25237, 25313, 25390, 25467, 25544, 25621, 25698, 25776, 25853, 25931,
    26009, 26086, 26165, 26243, 26321, 26399, 26478, 26557, 26635, 26714, 26793, 26872, 26952,
    27031, 27111, 27190, 27270, 27350, 27430, 27510, 27591, 27671, 27752, 27832, 27913, 27994,
    28075, 28157, 28238, 28319, 28401, 28483, 28565, 28647, 28729, 28811, 28893, 28976, 29059,
    29141, 29224, 29307, 29391, 29474, 29557, 29641, 29725, 29808, 29892, 29976, 30061, 30145,
    30229, 30314, 30399, 30484, 30569, 30654, 30739, 30824, 30910, 30995, 31081, 31167, 31253,
    31339, 31426, 31512, 31599, 31685, 31772, 31859, 31946, 32033, 32121, 32208, 32296, 32383,
    32471, 32559, 32647, 32736, 32824, 32913, 33001, 33090, 33179, 33268, 33357, 33446, 33536,
    33625, 33715, 33805, 33895, 33985, 34075, 34166, 34256, 34347, 34437, 34528, 34619, 34710,
    34802, 34893, 34985, 35076, 35168, 35260, 35352, 35444, 35537, 35629, 35722, 35814, 35907,
    36000, 36093, 36187, 36280, 36374, 36467, 36561, 36655, 36749, 36843, 36938, 37032, 37127,
    37221, 37316, 37411, 37506, 37601, 37697, 37792, 37888, 37984, 38080, 38176, 38272, 38368,
    38465, 38561, 38658, 38755, 38852, 38949, 39046, 39143, 39241, 39339, 39436, 39534, 39632,
    39731, 39829, 39927, 40026, 40125, 40223, 40322, 40422, 40521, 40620, 40720, 40819, 40919,
    41019, 41119, 41219, 41320, 41420, 41521, 41621, 41722, 41823, 41924, 42026, 42127, 42229,
    42330, 42432, 42534, 42636, 42738, 42840, 42943, 43046, 43148, 43251, 43354, 43457, 43561,
    43664, 43768, 43871, 43975, 44079, 44183, 44287, 44392, 44496, 44601, 44706, 44810, 44915,
    45021, 45126, 45231, 45337, 45443, 45549, 45655, 45761, 45867, 45973, 46080, 46186, 46293,
    46400, 46507, 46614, 46722, 46829, 46937, 47045, 47152, 47261, 47369, 47477, 47585, 47694,
    47803, 47911, 48020, 48130, 48239, 48348, 48458, 48567, 48677, 48787, 48897, 49007, 49118,
    49228, 49339, 49449, 49560, 49671, 49782, 49894, 50005, 50117, 50228, 50340, 50452, 50564,
    50677, 50789, 50902, 51014, 51127, 51240, 51353, 51466, 51579, 51693, 51807, 51920, 52034,
    52148, 52262, 52377, 52491, 52606, 52720, 52835, 52950, 53065, 53181, 53296, 53412, 53527,
    53643, 53759, 53875, 53991, 54108, 54224, 54341, 54458, 54575, 54692, 54809, 54926, 55044,
    55161, 55279, 55397, 55515, 55633, 55751, 55870, 55988, 56107, 56226, 56345, 56464, 56583,
    56703, 56822, 56942, 57062, 57182, 57302, 57422, 57542, 57663, 57784, 57904, 58025, 58146,
    58268, 58389, 58510, 58632, 58754, 58876, 58998, 59120, 59242, 59365, 59487, 59610, 59733,
    59856, 59979, 60102, 60226, 60349, 60473, 60597, 60721, 60845, 60969, 61094, 61218, 61343,
    61468, 61593, 61718, 61843, 61968, 62094, 62220, 62345, 62471, 62597, 62724, 62850, 62977,
    63103, 63230, 63357, 63484, 63611, 63738, 63866, 63994, 64121, 64249, 64377, 64505, 64634,
    64762, 64891, 65020, 65149, 65278, 65407, 65536, 65536,
];
pub(crate) const LINEAR_TO_GAMMA_TAB: [u32; 514] = [
    0, 576, 1152, 1728, 2304, 2880, 3456, 4032, 4608, 5184, 5751, 6288, 6799, 7287, 7755, 8204,
    8638, 9057, 9462, 9855, 10238, 10609, 10971, 11325, 11669, 12006, 12336, 12659, 12975, 13285,
    13589, 13888, 14182, 14470, 14754, 15033, 15308, 15578, 15845, 16108, 16367, 16623, 16875,
    17124, 17369, 17612, 17852, 18089, 18323, 18554, 18783, 19010, 19233, 19455, 19674, 19891,
    20106, 20319, 20530, 20739, 20946, 21151, 21354, 21555, 21755, 21952, 22149, 22343, 22536,
    22728, 22918, 23106, 23293, 23478, 23663, 23845, 24027, 24207, 24386, 24564, 24740, 24915,
    25089, 25262, 25434, 25604, 25774, 25942, 26109, 26276, 26441, 26605, 26768, 26931, 27092,
    27252, 27412, 27570, 27728, 27885, 28041, 28196, 28350, 28503, 28656, 28807, 28958, 29109,
    29258, 29407, 29555, 29702, 29848, 29994, 30139, 30283, 30427, 30570, 30712, 30854, 30995,
    31135, 31275, 31414, 31552, 31690, 31827, 31964, 32100, 32235, 32370, 32504, 32638, 32771,
    32904, 33036, 33167, 33298, 33429, 33559, 33688, 33817, 33946, 34074, 34201, 34328, 34455,
    34581, 34706, 34831, 34956, 35080, 35204, 35327, 35450, 35572, 35694, 35816, 35937, 36057,
    36178, 36298, 36417, 36536, 36655, 36773, 36891, 37008, 37125, 37242, 37358, 37474, 37590,
    37705, 37820, 37934, 38048, 38162, 38275, 38388, 38501, 38613, 38725, 38837, 38949, 39060,
    39170, 39281, 39391, 39500, 39610, 39719, 39827, 39936, 40044, 40152, 40259, 40367, 40474,
    40580, 40686, 40793, 40898, 41004, 41109, 41214, 41318, 41423, 41527, 41631, 41734, 41837,
    41940, 42043, 42145, 42248, 42350, 42451, 42553, 42654, 42755, 42855, 42956, 43056, 43156,
    43255, 43355, 43454, 43553, 43652, 43750, 43848, 43946, 44044, 44141, 44239, 44336, 44433,
    44529, 44626, 44722, 44818, 44913, 45009, 45104, 45199, 45294, 45389, 45483, 45578, 45672,
    45765, 45859, 45952, 46046, 46139, 46231, 46324, 46416, 46509, 46601, 46692, 46784, 46876,
    46967, 47058, 47149, 47239, 47330, 47420, 47510, 47600, 47690, 47780, 47869, 47958, 48047,
    48136, 48225, 48313, 48402, 48490, 48578, 48666, 48753, 48841, 48928, 49015, 49102, 49189,
    49276, 49362, 49448, 49535, 49621, 49706, 49792, 49878, 49963, 50048, 50133, 50218, 50303,
    50387, 50472, 50556, 50640, 50724, 50808, 50892, 50975, 51058, 51142, 51225, 51308, 51390,
    51473, 51556, 51638, 51720, 51802, 51884, 51966, 52048, 52129, 52210, 52292, 52373, 52454,
    52535, 52615, 52696, 52776, 52857, 52937, 53017, 53097, 53176, 53256, 53335, 53415, 53494,
    53573, 53652, 53731, 53810, 53888, 53967, 54045, 54124, 54202, 54280, 54357, 54435, 54513,
    54590, 54668, 54745, 54822, 54899, 54976, 55053, 55130, 55206, 55283, 55359, 55435, 55511,
    55587, 55663, 55739, 55815, 55890, 55965, 56041, 56116, 56191, 56266, 56341, 56416, 56490,
    56565, 56639, 56714, 56788, 56862, 56936, 57010, 57084, 57157, 57231, 57305, 57378, 57451,
    57524, 57598, 57670, 57743, 57816, 57889, 57961, 58034, 58106, 58179, 58251, 58323, 58395,
    58467, 58538, 58610, 58682, 58753, 58825, 58896, 58967, 59038, 59109, 59180, 59251, 59322,
    59393, 59463, 59534, 59604, 59674, 59744, 59815, 59885, 59954, 60024, 60094, 60164, 60233,
    60303, 60372, 60441, 60511, 60580, 60649, 60718, 60787, 60855, 60924, 60993, 61061, 61130,
    61198, 61266, 61334, 61403, 61471, 61539, 61606, 61674, 61742, 61809, 61877, 61944, 62012,
    62079, 62146, 62213, 62280, 62347, 62414, 62481, 62548, 62614, 62681, 62747, 62814, 62880,
    62946, 63013, 63079, 63145, 63211, 63277, 63342, 63408, 63474, 63539, 63605, 63670, 63736,
    63801, 63866, 63931, 63996, 64061, 64126, 64191, 64256, 64320, 64385, 64450, 64514, 64578,
    64643, 64707, 64771, 64835, 64899, 64963, 65027, 65091, 65155, 65219, 65282, 65346, 65409,
    65473, 65536, 65536,
];

//------------------------------------------------------------------------------
// Gamma (sharpyuv_gamma.c, sRGB path). The tables above are the exact values
// libwebp computes in `SharpYuvInitGammaTables` (dumped from an instrumented
// build — see `benchmarks/sharpyuv_port_2026-07-16.md`), so the port cannot
// drift from libwebp via libm `pow` differences.

const GAMMA_TO_LINEAR_TAB_BITS: u32 = 10;
const LINEAR_TO_GAMMA_TAB_BITS: u32 = 9;
const GAMMA_TO_LINEAR_BITS: u32 = 16;

/// `FixedPointInterpolation` — linear interpolation between adjacent table
/// entries in `tab_pos_shift_right` fixed-point precision.
#[inline]
fn fixed_point_interpolation(
    v: u32,
    tab: &[u32],
    tab_pos_shift_right: u32,
    tab_value_shift_right: u32,
) -> u32 {
    let tab_pos = (v >> tab_pos_shift_right) as usize;
    // Fractional part, in 'tab_pos_shift' fixed-point precision.
    let x = v - ((tab_pos as u32) << tab_pos_shift_right);
    let v0 = tab[tab_pos] >> tab_value_shift_right;
    let v1 = tab[tab_pos + 1] >> tab_value_shift_right;
    let v2 = (v1 - v0) * x; // note: v1 >= v0 (tables are monotone)
    let half = if tab_pos_shift_right > 0 {
        1u32 << (tab_pos_shift_right - 1)
    } else {
        0
    };
    v0 + ((v2 + half) >> tab_pos_shift_right)
}

/// `ToLinearSrgb` at bit_depth = 10: `GAMMA_TO_LINEAR_TAB_BITS - 10 == 0`,
/// so the interpolation degenerates to a direct table lookup.
#[inline]
fn gamma_to_linear(v: u16) -> u32 {
    debug_assert_eq!(GAMMA_TO_LINEAR_TAB_BITS, Y_BIT_DEPTH);
    GAMMA_TO_LINEAR_TAB[v as usize]
}

/// `FromLinearSrgb` at bit_depth = 10: interpolate the 9-bit table, values
/// scaled from 16-bit to 10-bit fixed point.
#[inline]
fn linear_to_gamma(value: u32) -> u32 {
    fixed_point_interpolation(
        value,
        &LINEAR_TO_GAMMA_TAB,
        GAMMA_TO_LINEAR_BITS - LINEAR_TO_GAMMA_TAB_BITS,
        GAMMA_TO_LINEAR_BITS - Y_BIT_DEPTH,
    )
}

//------------------------------------------------------------------------------
// Pixel helpers (sharpyuv.c)

#[inline]
fn clip_8b(v: i32) -> u8 {
    if v & !0xff == 0 {
        v as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

#[inline]
fn clip_bit_depth(y: i32) -> u16 {
    if y & !MAX_Y == 0 {
        y as u16
    } else if y < 0 {
        0
    } else {
        MAX_Y as u16
    }
}

/// BT.709 luma in YUV_FIX fixed point. i64 because `UpdateW` feeds 16-bit
/// fixed-point linear values (≤ 65536), where the weighted sum exceeds i32.
#[inline]
fn rgb_to_gray(r: i64, g: i64, b: i64) -> i32 {
    ((13933 * r + 46871 * g + 4732 * b + K_YUV_HALF) >> YUV_FIX) as i32
}

/// Average four gamma-space samples in LINEAR light.
#[inline]
fn scale_down(a: u16, b: u16, c: u16, d: u16) -> u32 {
    let (a, b, c, d) = (
        gamma_to_linear(a),
        gamma_to_linear(b),
        gamma_to_linear(c),
        gamma_to_linear(d),
    );
    linear_to_gamma((a + b + c + d + 2) >> 2)
}

/// `UpdateW`: per-pixel target luma of an R,G,B row triple, via linear light.
/// `src` holds three `w`-length channel rows (R at 0, G at w, B at 2w).
fn update_w(src: &[u16], dst: &mut [u16], w: usize) {
    for i in 0..w {
        let r = gamma_to_linear(src[i]);
        let g = gamma_to_linear(src[w + i]);
        let b = gamma_to_linear(src[2 * w + i]);
        let y = rgb_to_gray(i64::from(r), i64::from(g), i64::from(b));
        dst[i] = linear_to_gamma(y as u32) as u16;
    }
}

/// `UpdateChroma`: per-2×2-block target chroma deltas (r−W, g−W, b−W),
/// downsampling each channel in linear light.
fn update_chroma(src1: &[u16], src2: &[u16], dst: &mut [i16], uv_w: usize) {
    for i in 0..uv_w {
        let r = scale_down(src1[2 * i], src1[2 * i + 1], src2[2 * i], src2[2 * i + 1]) as i32;
        let g = scale_down(
            src1[2 * uv_w + 2 * i],
            src1[2 * uv_w + 2 * i + 1],
            src2[2 * uv_w + 2 * i],
            src2[2 * uv_w + 2 * i + 1],
        ) as i32;
        let b = scale_down(
            src1[4 * uv_w + 2 * i],
            src1[4 * uv_w + 2 * i + 1],
            src2[4 * uv_w + 2 * i],
            src2[4 * uv_w + 2 * i + 1],
        ) as i32;
        let w = rgb_to_gray(i64::from(r), i64::from(g), i64::from(b));
        dst[i] = (r - w) as i16;
        dst[uv_w + i] = (g - w) as i16;
        dst[2 * uv_w + i] = (b - w) as i16;
    }
}

/// `StoreGray`: initial W plane = gray of the gamma-space R,G,B rows.
fn store_gray(rgb: &[u16], y: &mut [u16], w: usize) {
    for i in 0..w {
        y[i] = rgb_to_gray(
            i64::from(rgb[i]),
            i64::from(rgb[w + i]),
            i64::from(rgb[2 * w + i]),
        ) as u16;
    }
}

/// `Filter2`: boundary tap of the 9-3-3-1 filter, (3A + B + 2) >> 2.
#[inline]
fn filter2(a: i32, b: i32, w0: i32) -> u16 {
    let v0 = (a * 3 + b + 2) >> 2;
    clip_bit_depth(v0 + w0)
}

//------------------------------------------------------------------------------
// DSP kernels (sharpyuv_dsp.c, C reference — bit-equal to the SSE2 build).

/// `SharpYuvUpdateY_C`: move the current luma toward the target, returning
/// the summed absolute luma error BEFORE the update.
fn sharp_yuv_update_y(target: &[u16], cur: &[u16], dst: &mut [u16], len: usize) -> u64 {
    let mut diff: u64 = 0;
    for i in 0..len {
        let diff_y = i32::from(target[i]) - i32::from(cur[i]);
        let new_y = i32::from(dst[i]) + diff_y;
        dst[i] = clip_bit_depth(new_y);
        diff += u64::from(diff_y.unsigned_abs());
    }
    diff
}

/// `SharpYuvUpdateRGB_C`: move the chroma deltas toward the target.
fn sharp_yuv_update_rgb(target: &[i16], cur: &[i16], dst: &mut [i16], len: usize) {
    for i in 0..len {
        let diff_uv = i32::from(target[i]) - i32::from(cur[i]);
        dst[i] = (i32::from(dst[i]) + diff_uv) as i16;
    }
}

/// `SharpYuvFilterRow_C`: 9-3-3-1 upsampling of the chroma delta rows A
/// (current) and B (neighbor), added to the luma plane and clipped. This is
/// the same filter the decoder's fancy upsampler applies, which is the whole
/// point: the optimization loop sees exactly what the decoder will produce.
fn sharp_yuv_filter_row(a: &[i16], b: &[i16], len: usize, best_y: &[u16], out: &mut [u16]) {
    for i in 0..len {
        let a0 = i32::from(a[i]);
        let a1 = i32::from(a[i + 1]);
        let b0 = i32::from(b[i]);
        let b1 = i32::from(b[i + 1]);
        let v0 = (a0 * 9 + a1 * 3 + b0 * 3 + b1 + 8) >> 4;
        let v1 = (a1 * 9 + a0 * 3 + b1 * 3 + b0 + 8) >> 4;
        out[2 * i] = clip_bit_depth(i32::from(best_y[2 * i]) + v0);
        out[2 * i + 1] = clip_bit_depth(i32::from(best_y[2 * i + 1]) + v1);
    }
}

//------------------------------------------------------------------------------
// Row import / interpolation (sharpyuv.c)

/// `ImportOneRow` (8-bit): interleaved row → three channel rows shifted to
/// 10-bit, right-replicating the last pixel when the width is odd.
fn import_one_row(
    row: &[u8],
    step: usize,
    r_off: usize,
    g_off: usize,
    b_off: usize,
    pic_width: usize,
    dst: &mut [u16],
) {
    let w = (pic_width + 1) & !1;
    for i in 0..pic_width {
        let off = i * step;
        dst[i] = u16::from(row[off + r_off]) << SFIX;
        dst[w + i] = u16::from(row[off + g_off]) << SFIX;
        dst[2 * w + i] = u16::from(row[off + b_off]) << SFIX;
    }
    if pic_width & 1 != 0 {
        dst[pic_width] = dst[pic_width - 1];
        dst[w + pic_width] = dst[w + pic_width - 1];
        dst[2 * w + pic_width] = dst[2 * w + pic_width - 1];
    }
}

/// `InterpolateTwoRows`: reconstruct two full-resolution R,G,B rows from the
/// luma plane plus 9-3-3-1-upsampled chroma deltas (prev/cur/next uv rows).
#[allow(clippy::too_many_arguments)]
fn interpolate_two_rows(
    best_y: &[u16],
    uv_buf: &[i16],
    prev_off: usize,
    cur_off: usize,
    next_off: usize,
    w: usize,
    out1: &mut [u16],
    out2: &mut [u16],
) {
    let uv_w = w >> 1;
    let len = (w - 1) >> 1; // length to filter
    for k in 0..3 {
        // Process each R/G/B segment in turn.
        let prev = &uv_buf[prev_off + k * uv_w..prev_off + (k + 1) * uv_w];
        let cur = &uv_buf[cur_off + k * uv_w..cur_off + (k + 1) * uv_w];
        let next = &uv_buf[next_off + k * uv_w..next_off + (k + 1) * uv_w];
        let o1 = &mut out1[k * w..(k + 1) * w];
        let o2 = &mut out2[k * w..(k + 1) * w];

        // Special boundary case for i == 0.
        o1[0] = filter2(i32::from(cur[0]), i32::from(prev[0]), i32::from(best_y[0]));
        o2[0] = filter2(i32::from(cur[0]), i32::from(next[0]), i32::from(best_y[w]));

        sharp_yuv_filter_row(cur, prev, len, &best_y[1..], &mut o1[1..]);
        sharp_yuv_filter_row(cur, next, len, &best_y[w + 1..], &mut o2[1..]);

        // Special boundary case for i == w - 1 when w is even.
        if w & 1 == 0 {
            o1[w - 1] = filter2(
                i32::from(cur[uv_w - 1]),
                i32::from(prev[uv_w - 1]),
                i32::from(best_y[w - 1]),
            );
            o2[w - 1] = filter2(
                i32::from(cur[uv_w - 1]),
                i32::from(next[uv_w - 1]),
                i32::from(best_y[w - 1 + w]),
            );
        }
    }
}

//------------------------------------------------------------------------------
// Final W/RGB → YUV conversion (sharpyuv.c)

/// `RGBToYUVComponent` with the SFIX-scaled matrix row.
#[inline]
fn rgb_to_yuv_component(r: i32, g: i32, b: i32, coeffs: &[i32; 4]) -> i32 {
    let srounder = 1 << (YUV_FIX + SFIX - 1);
    let luma = coeffs[0] * r + coeffs[1] * g + coeffs[2] * b + coeffs[3] + srounder;
    luma >> (YUV_FIX + SFIX)
}

/// `ConvertWRGBToYUV` for tightly-packed 8-bit output planes.
fn convert_wrgb_to_yuv(
    best_y: &[u16],
    best_uv: &[i16],
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = (width + 1) & !1;
    let h = (height + 1) & !1;
    let uv_w = w >> 1;
    let uv_h = h >> 1;
    let mut y_out = vec![0u8; width * height];
    let mut u_out = vec![0u8; uv_w * uv_h];
    let mut v_out = vec![0u8; uv_w * uv_h];

    for j in 0..height {
        let y_row = &best_y[j * w..];
        let uv_row = &best_uv[(j >> 1) * 3 * uv_w..];
        let out = &mut y_out[j * width..(j + 1) * width];
        for (i, out_px) in out.iter_mut().enumerate() {
            let off = i >> 1;
            let w_val = i32::from(y_row[i]);
            let r = i32::from(uv_row[off]) + w_val;
            let g = i32::from(uv_row[uv_w + off]) + w_val;
            let b = i32::from(uv_row[2 * uv_w + off]) + w_val;
            *out_px = clip_8b(rgb_to_yuv_component(r, g, b, &RGB_TO_Y));
        }
    }

    for j in 0..uv_h {
        let uv_row = &best_uv[j * 3 * uv_w..];
        let u_row = &mut u_out[j * uv_w..(j + 1) * uv_w];
        let v_row = &mut v_out[j * uv_w..(j + 1) * uv_w];
        for i in 0..uv_w {
            // r, g, b are off by W here, but a constant offset on all three
            // components doesn't change U or V with a YCbCr matrix.
            let r = i32::from(uv_row[i]);
            let g = i32::from(uv_row[uv_w + i]);
            let b = i32::from(uv_row[2 * uv_w + i]);
            u_row[i] = clip_8b(rgb_to_yuv_component(r, g, b, &RGB_TO_U));
            v_row[i] = clip_8b(rgb_to_yuv_component(r, g, b, &RGB_TO_V));
        }
    }
    (y_out, u_out, v_out)
}

//------------------------------------------------------------------------------
// Main loop (`DoSharpArgbToYuv`)

/// libwebp `SharpYuvConvert` for 8-bit interleaved RGB/RGBA/BGR/BGRA input,
/// returning tightly-packed Y (width×height) and U/V (⌈w/2⌉×⌈h/2⌉) planes.
/// Byte-identical to libwebp at the WebP-encoder operating point.
pub(crate) fn sharp_yuv_convert(
    rgb: &[u8],
    step: usize,
    r_off: usize,
    g_off: usize,
    b_off: usize,
    width: usize,
    height: usize,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    assert!(width > 0 && height > 0);
    // We expand the right/bottom border if needed.
    let w = (width + 1) & !1;
    let h = (height + 1) & !1;
    let uv_w = w >> 1;
    let uv_h = h >> 1;

    let mut tmp_buffer = vec![0u16; w * 3 * 2]; // src1 | src2
    let mut best_y = vec![0u16; w * h];
    let mut target_y = vec![0u16; w * h];
    let mut best_rgb_y = vec![0u16; w * 2];
    let mut best_uv = vec![0i16; uv_w * 3 * uv_h];
    let mut target_uv = vec![0i16; uv_w * 3 * uv_h];
    let mut best_rgb_uv = vec![0i16; uv_w * 3];
    // NB: libwebp derives the threshold from the PADDED dims (its local
    // w/h are already rounded up to even) — using width*height here changes
    // the iteration count on odd-dimension images.
    let diff_y_threshold = (3.0 * w as f64 * h as f64) as u64;

    // Import RGB samples to W/RGB representation.
    for j in (0..height).step_by(2) {
        let is_last_row = j == height - 1;
        let (src1, src2) = tmp_buffer.split_at_mut(3 * w);
        import_one_row(&rgb[j * stride..], step, r_off, g_off, b_off, width, src1);
        if !is_last_row {
            import_one_row(
                &rgb[(j + 1) * stride..],
                step,
                r_off,
                g_off,
                b_off,
                width,
                src2,
            );
        } else {
            src2.copy_from_slice(src1);
        }
        let y_base = (j & !1) * w;
        let uv_base = (j >> 1) * 3 * uv_w;
        store_gray(src1, &mut best_y[y_base..], w);
        store_gray(src2, &mut best_y[y_base + w..], w);
        update_w(src1, &mut target_y[y_base..], w);
        update_w(src2, &mut target_y[y_base + w..], w);
        update_chroma(
            src1,
            src2,
            &mut target_uv[uv_base..uv_base + 3 * uv_w],
            uv_w,
        );
        best_uv[uv_base..uv_base + 3 * uv_w]
            .copy_from_slice(&target_uv[uv_base..uv_base + 3 * uv_w]);
    }

    // Iterate and resolve clipping conflicts.
    let mut prev_diff_y_sum = u64::MAX;
    for iter in 0..K_NUM_ITERATIONS {
        let mut prev_off = 0usize;
        let mut cur_off = 0usize;
        let mut diff_y_sum: u64 = 0;

        let mut j = 0usize;
        while j < h {
            let next_off = cur_off + if j < h - 2 { 3 * uv_w } else { 0 };
            {
                let (src1, src2) = tmp_buffer.split_at_mut(3 * w);
                interpolate_two_rows(
                    &best_y[j * w..],
                    &best_uv,
                    prev_off,
                    cur_off,
                    next_off,
                    w,
                    src1,
                    src2,
                );
            }
            prev_off = cur_off;
            cur_off = next_off;

            let (src1, src2) = tmp_buffer.split_at(3 * w);
            update_w(src1, &mut best_rgb_y[..w], w);
            update_w(src2, &mut best_rgb_y[w..], w);
            update_chroma(src1, src2, &mut best_rgb_uv, uv_w);

            // Update two rows of Y and one row of RGB.
            let uv_base = (j >> 1) * 3 * uv_w;
            diff_y_sum += sharp_yuv_update_y(
                &target_y[j * w..(j + 2) * w],
                &best_rgb_y,
                &mut best_y[j * w..(j + 2) * w],
                2 * w,
            );
            sharp_yuv_update_rgb(
                &target_uv[uv_base..uv_base + 3 * uv_w],
                &best_rgb_uv,
                &mut best_uv[uv_base..uv_base + 3 * uv_w],
                3 * uv_w,
            );
            j += 2;
        }
        // Test exit condition.
        if iter > 0 {
            if diff_y_sum < diff_y_threshold {
                break;
            }
            if diff_y_sum > prev_diff_y_sum {
                break;
            }
        }
        prev_diff_y_sum = diff_y_sum;
    }

    // Final reconstruction.
    convert_wrgb_to_yuv(&best_y, &best_uv, width, height)
}

/// Layout-aware wrapper matching `convert_image_yuv_fast`'s interface:
/// converts via the SharpYUV port, then pads the tight planes to the
/// MB-aligned dimensions the encoder consumes (same edge replication as the
/// standard conversion paths).
pub(crate) fn convert_image_sharp_libwebp(
    image_data: &[u8],
    color: crate::encoder::PixelLayout,
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    use crate::encoder::PixelLayout;
    let (step, r_off, g_off, b_off) = match color {
        PixelLayout::Rgb8 => (3, 0, 1, 2),
        PixelLayout::Rgba8 => (4, 0, 1, 2),
        PixelLayout::Bgr8 => (3, 2, 1, 0),
        PixelLayout::Bgra8 => (4, 2, 1, 0),
        PixelLayout::L8 | PixelLayout::La8 | PixelLayout::Yuv420 | PixelLayout::Argb8 => {
            unreachable!("sharp YUV parity conversion requires RGB/BGR input")
        }
    };
    let (w, h) = (usize::from(width), usize::from(height));
    // `stride` is in PIXELS at this interface (matching
    // `convert_image_yuv_fast`); the port wants a byte stride.
    let (y, u, v) = sharp_yuv_convert(image_data, step, r_off, g_off, b_off, w, h, stride * step);

    let mb_w = w.div_ceil(16);
    let mb_h = h.div_ceil(16);
    let (luma_w, luma_h) = (16 * mb_w, 16 * mb_h);
    let (chroma_w, chroma_h) = (8 * mb_w, 8 * mb_h);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut y_out = vec![0u8; luma_w * luma_h];
    let mut u_out = vec![0u8; chroma_w * chroma_h];
    let mut v_out = vec![0u8; chroma_w * chroma_h];
    crate::decoder::yuv::pad_plane(&y, &mut y_out, w, h, luma_w, luma_h);
    crate::decoder::yuv::pad_plane(&u, &mut u_out, cw, ch, chroma_w, chroma_h);
    crate::decoder::yuv::pad_plane(&v, &mut v_out, cw, ch, chroma_w, chroma_h);
    (y_out, u_out, v_out)
}
