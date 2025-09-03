#include "mex.h"
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Blink_C_wrapper.h"      // 确保与你的安装路径一致
#include <windows.h>

static inline double gs(const mxArray* a) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a)!=1)
        mexErrMsgIdAndTxt("SLM:Type","All scalars must be real double.");
    return *mxGetPr(a);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 13)
        mexErrMsgIdAndTxt("SLM:Args",
          "Usage: rc = slm_fresnel_write_mex(board,width,height,focal_mm,ax_mrad,ay_mrad,"
          " [phase_scale],[wavelength],[pixel_size], waitTrig, flipImm, outFlip, outRefresh, timeout_ms)");

    int   board  = (int)gs(prhs[0]);
    int   width  = (int)gs(prhs[1]);
    int   height = (int)gs(prhs[2]);
    float focal_mm     = (float)gs(prhs[3]);
    float ax_mrad      = (float)gs(prhs[4]);
    float ay_mrad      = (float)gs(prhs[5]);
    float phase_scale  = (nrhs >= 7) ? (float)gs(prhs[6]) : 1.0f;
    float wavelength   = (nrhs >= 8) ? (float)gs(prhs[7]) : 5.50e-7f;
    float pixel_sz     = (nrhs >= 9) ? (float)gs(prhs[8]) : 9.2e-6f;

    int waitTrig = (int)gs(prhs[9]);
    int flipImm  = (int)gs(prhs[10]);
    int outFlip  = (int)gs(prhs[11]);
    int outRefr  = (int)gs(prhs[12]);
    int timeout  = (nrhs >= 14) ? (int)gs(prhs[13]) : 5000;

    if (width<=0 || height<=0) mexErrMsgIdAndTxt("SLM:Size","width/height must be >0.");

    // —— 生成 Fresnel + 倾斜，相位直接填“行主序”一维缓冲，以匹配 Write_image —— //
    const float TWO_PI = 6.283185307179586f;
    const float f_m    = focal_mm * 1e-3f;
    const float k      = TWO_PI / wavelength;
    const float f2     = f_m * f_m;

    const float cx = (width  - 1) * 0.5f;   // 用0-based像素中心
    const float cy = (height - 1) * 0.5f;

    const float gx = k * std::sinf(ax_mrad * 1e-3f);
    const float gy = k * std::sinf(ay_mrad * 1e-3f);

    const float scale = 255.0f / TWO_PI;
    const float ck    = scale * k;
    const float gx_s  = scale * gx;
    const float gy_s  = scale * gy;

    std::vector<float> y(height), y2(height);
    for (int r=0; r<height; ++r) {
        float yr = ( (float)r - cy ) * pixel_sz;
        y[r]  = yr;
        y2[r] = yr*yr;
    }

    std::vector<unsigned char> buf((size_t)width * (size_t)height);

    #pragma omp parallel for schedule(static)
    for (int r=0; r<height; ++r) {
        for (int c=0; c<width; ++c) {
            float xm = ( (float)c - cx ) * pixel_sz;
            float r2 = xm*xm + y2[r];
            float phs_scaled = ck*(f_m - std::sqrt(f2 + r2)) + gx_s*xm + gy_s*y[r];
            float m = std::fmod(phs_scaled, 256.0f) * phase_scale;
            if (m < 0.0f) m += 256.0f;
            buf[(size_t)r*width + c] = (unsigned char)m; // 行主序
        }
    }

    // —— 调用 SDK 的 Write_image —— //
    // 方案A：已链接 .lib，直接调用
    int rc = Write_image(board, buf.data(), (int)buf.size(),
                         waitTrig, flipImm, outFlip, outRefr, timeout);

    // 可选：输出返回码
    if (nlhs >= 1) {
        plhs[0] = mxCreateDoubleScalar((double)rc);
    }
}
