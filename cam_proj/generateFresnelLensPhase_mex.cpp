#include "mex.h"
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// 安全取标量
static inline double getScalar(const mxArray* a) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a)!=1)
        mexErrMsgIdAndTxt("FresnelMex:Type","All scalars must be real double.");
    return *mxGetPr(a);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 5 || nrhs > 8)
        mexErrMsgIdAndTxt("FresnelMex:Args",
            "Usage: out = generateFresnelLensPhase_mex(width,height,focal_mm,angle_x_mrad,angle_y_mrad,[wavelength],[pixel_size])");

    const int width  = static_cast<int>(getScalar(prhs[0]));
    const int height = static_cast<int>(getScalar(prhs[1]));
    if (width <= 0 || height <= 0) mexErrMsgIdAndTxt("FresnelMex:Size","width/height must be >0.");

    const float focal_mm      = static_cast<float>(getScalar(prhs[2]));
    const float angle_x_mrad  = static_cast<float>(getScalar(prhs[3]));
    const float angle_y_mrad  = static_cast<float>(getScalar(prhs[4]));
    const float phase_scale  = (nrhs >= 6) ? static_cast<float>(getScalar(prhs[5])) : 1.0f;
    const float wavelength    = (nrhs >= 7) ? static_cast<float>(getScalar(prhs[6])) : 5.50e-7f;
    const float pixel_size    = (nrhs == 8) ? static_cast<float>(getScalar(prhs[7])) : 9.2e-6f;

    // 输出: height x width, uint8
    plhs[0] = mxCreateNumericMatrix(height, width, mxUINT8_CLASS, mxREAL);
    unsigned char* out = static_cast<unsigned char*>(mxGetData(plhs[0]));

    const float TWO_PI = 6.2831853071795864769f;
    const float f_m    = focal_mm * 1e-3f;
    const float k      = TWO_PI / wavelength;
    const float f2     = f_m * f_m;

    const float cx = (static_cast<float>(width)  + 1.f) * 0.5f;
    const float cy = (static_cast<float>(height) + 1.f) * 0.5f;

    // 梯度（只算一次）
    const float gx = k * std::sin(angle_x_mrad * 1e-3f);
    const float gy = k * std::sin(angle_y_mrad * 1e-3f);

    // 合并常数，减少乘法
    const float scale = 255.0f / TWO_PI * phase_scale;
    const float ck    = scale * k;
    const float gx_s  = scale * gx;
    const float gy_s  = scale * gy;

    // 预计算每行 y 与 y^2
    std::vector<float> y(height), y2(height);
    for (int r = 0; r < height; ++r) {
        const float yr = (static_cast<float>(r+1) - cy) * pixel_size;
        y[r]  = yr;
        y2[r] = yr*yr;
    }

    // 并行列循环（列主序，行做内层，可连续写 out）
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < width; ++c) {
        const float xm = (static_cast<float>(c+1) - cx) * pixel_size;
        const float x2 = xm * xm;
        const size_t col_off = static_cast<size_t>(c) * static_cast<size_t>(height);

        for (int r = 0; r < height; ++r) {
            const float r2 = x2 + y2[r];
            // 缩放后的相位
            float phs_scaled = ck*(f_m - std::sqrt(f2 + r2)) + gx_s*xm + gy_s*y[r];

            // mod 256 到 [0,256)
            float m = std::fmod(phs_scaled, 256.0f);
            if (m < 0.0f) m += 256.0f;

            out[col_off + r] = static_cast<unsigned char>(m);
        }
    }
}
