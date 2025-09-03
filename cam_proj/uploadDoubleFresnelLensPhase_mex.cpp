#include "mex.h"
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "Blink_C_wrapper.h" // 确保与你的安装路径一致
#include <windows.h>

static inline double gs(const mxArray* a) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a)!=1)
        mexErrMsgIdAndTxt("SLM:Type","All scalars must be real double.");
    return mxGetPr(a)[0];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray* prhs[]) {
    // 修改参数列表：现在需要4个倾斜参数
    if (nrhs < 15)
        mexErrMsgIdAndTxt("SLM:Args",
            "Usage: rc = slm_dual_fresnel_write_mex(board,width,height,focal_mm,"
            "ax_left_mrad,ay_left_mrad,ax_right_mrad,ay_right_mrad,"
            "[phase_scale],[wavelength],[pixel_size], waitTrig, flipImm, outFlip, outRefresh, timeout_ms)");
    
    int board = (int)gs(prhs[0]);
    int width = (int)gs(prhs[1]);
    int height = (int)gs(prhs[2]);
    float focal_mm = (float)gs(prhs[3]);
    
    // 四个独立的倾斜控制参数
    float ax_left_mrad = (float)gs(prhs[4]);
    float ay_left_mrad = (float)gs(prhs[5]);
    float ax_right_mrad = (float)gs(prhs[6]);
    float ay_right_mrad = (float)gs(prhs[7]);
    
    float phase_scale = (nrhs >= 9) ? (float)gs(prhs[8]) : 1.0f;
    float wavelength = (nrhs >= 10) ? (float)gs(prhs[9]) : 5.50e-7f;
    float pixel_sz = (nrhs >= 11) ? (float)gs(prhs[10]) : 9.2e-6f;
    
    int waitTrig = (int)gs(prhs[11]);
    int flipImm = (int)gs(prhs[12]);
    int outFlip = (int)gs(prhs[13]);
    int outRefr = (int)gs(prhs[14]);
    int timeout = (nrhs >= 16) ? (int)gs(prhs[15]) : 5000;
    
    if (width<=0 || height<=0) 
        mexErrMsgIdAndTxt("SLM:Size","width/height must be >0.");
    
    // —— 生成双Fresnel + 倾斜，相位直接填"行主序"一维缓冲 —— //
    const float TWO_PI = 6.283185307179586f;
    const float f_m = focal_mm * 1e-3f;
    const float k = TWO_PI / wavelength;
    const float f2 = f_m * f_m;
    
    // 计算左右两个半区的中心点
    int width_half = width / 2;
    const float cx_left = (width_half - 1) * 0.5f;  // 左半部分的中心
    const float cx_right = width_half + (width - width_half - 1) * 0.5f;  // 右半部分的中心
    const float cy = (height - 1) * 0.5f;  // y方向中心保持不变
    
    // 左右两侧的倾斜参数
    const float gx_left = k * std::sinf(ax_left_mrad * 1e-3f);
    const float gy_left = k * std::sinf(ay_left_mrad * 1e-3f);
    const float gx_right = k * std::sinf(ax_right_mrad * 1e-3f);
    const float gy_right = k * std::sinf(ay_right_mrad * 1e-3f);
    
    const float scale = 255.0f / TWO_PI;
    const float ck = scale * k;
    const float gx_left_s = scale * gx_left;
    const float gy_left_s = scale * gy_left;
    const float gx_right_s = scale * gx_right;
    const float gy_right_s = scale * gy_right;
    
    // 预计算y坐标相关值
    std::vector<float> y(height), y2_left(height), y2_right(height);
    for (int r=0; r<height; ++r) {
        float yr = ((float)r - cy) * pixel_sz;
        y[r] = yr;
        y2_left[r] = yr * yr;   // 左侧用
        y2_right[r] = yr * yr;  // 右侧用（实际上相同，但为清晰分开）
    }
    
    std::vector<unsigned char> buf((size_t)width * (size_t)height);
    
    #pragma omp parallel for schedule(static)
    for (int r=0; r<height; ++r) {
        for (int c=0; c<width; ++c) {
            float xm, cx_use;
            float gx_s_use, gy_s_use;
            
            // 判断像素位于左半部分还是右半部分
            if (c < width_half) {
                // 左半部分：使用左侧的中心和倾斜参数
                cx_use = cx_left;
                gx_s_use = gx_left_s;
                gy_s_use = gy_left_s;
            } else {
                // 右半部分：使用右侧的中心和倾斜参数
                cx_use = cx_right;
                gx_s_use = gx_right_s;
                gy_s_use = gy_right_s;
            }
            
            // 计算相对于对应中心的坐标
            xm = ((float)c - cx_use) * pixel_sz;
            
            // 计算Fresnel相位 + 倾斜相位
            float r2 = xm*xm + y2_left[r];  // y2_left和y2_right相同
            float phs_scaled = ck*(f_m - std::sqrt(f2 + r2)) + gx_s_use*xm + gy_s_use*y[r];
            
            // 相位缩放和范围调整
            float m = std::fmod(phs_scaled, 256.0f) * phase_scale;
            if (m < 0.0f) m += 256.0f;
            
            buf[(size_t)r*width + c] = (unsigned char)m; // 行主序
        }
    }
    
    // —— 调用 SDK 的 Write_image —— //
    int rc = Write_image(board, buf.data(), (int)buf.size(),
                        waitTrig, flipImm, outFlip, outRefr, timeout);
    
    // 可选：输出返回码
    if (nlhs >= 1) {
        plhs[0] = mxCreateDoubleScalar((double)rc);
    }
}