%% init camera
% % install webcam add-on
% matlab.internal.addons.showAddon('USBWEBCAM');

% Create the webcam object.
cam = webcam('Elgato HD60 X');
cam.Resolution = '1280x720';
videoFrame = snapshot(cam);
frameSize = size(videoFrame);
% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
%%  init SLM
addpath("C:\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\")
if ~libisloaded('Blink_C_wrapper')
    loadlibrary('Blink_C_wrapper.dll', 'Blink_C_wrapper.h');
end
% This loads the image generation functions
if ~libisloaded('ImageGen')
    loadlibrary('ImageGen.dll', 'ImageGen.h');
end
% Basic parameters for calling Create_SDK
bit_depth = 12; %bit depth = 8 for small 512, 12 for 1920
num_boards_found = libpointer('uint32Ptr', 0);
constructed_okay = libpointer('int32Ptr', 0);
is_nematic_type = 1;
RAM_write_enable = 1;
use_GPU = 1;
max_transients = 5;
wait_For_Trigger = 0; % This feature is user-settable; use 1 for 'on' or 0 for 'off'
flip_immediate = 0; % Only supported on the 1024
timeout_ms = 5000;
RGB = 0;
%Both pulse options can be false, but only one can be true. You either generate a pulse when the new image begins loading to the SLM
%or every 1.184 ms on SLM refresh boundaries, or if both are false no output pulse is generated.
OutputPulseImageFlip = 0;
OutputPulseImageRefresh = 0; %only supported on 1920x1152, FW rev 1.8. 
% - This regional LUT file is only used with Overdrive Plus, otherwise it should always be a null string
reg_lut = libpointer('string');
% Call the constructor
calllib('Blink_C_wrapper', 'Create_SDK', bit_depth, num_boards_found, constructed_okay, is_nematic_type, RAM_write_enable, use_GPU, max_transients, reg_lut);
% constructed okay return of 1 is success
if constructed_okay.value ~= 1  
    disp(calllib('Blink_C_wrapper', 'Get_last_error_message'));
end
board_number = 1;
disp('Blink SDK was successfully constructed');
fprintf('Found %u SLM controller(s)\n', num_boards_found.value);
height = calllib('Blink_C_wrapper', 'Get_image_height', board_number);
width = calllib('Blink_C_wrapper', 'Get_image_width', board_number);
depth = calllib('Blink_C_wrapper', 'Get_image_depth', board_number); %bits per pixel
Bytes = depth/8;
%but for now open a generic LUT that linearly maps input graylevels to output voltages
%***Using *_linearVoltage.LUT does NOT give a linear phase response***

if ((width == 512) && (depth == 8))
	calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\512x512_linearVoltage.LUT');
end
if ((width == 512) && (depth == 16))
	calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\512x512_16bit_linearVoltage.LUT');
end
if width == 1920
	% calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1920x1152_linearVoltage.LUT');
    calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm5691_at635');
end
if width == 1024
	calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, 'C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT');
end
%allocate arrays for our images
ImageOne = libpointer('uint8Ptr', zeros(width*height*Bytes,1));
WFC = libpointer('uint8Ptr', zeros(width*height*Bytes,1));
% Start the SLM with a blank image
calllib('Blink_C_wrapper', 'Write_image', board_number, ImageOne, width*height*Bytes, wait_For_Trigger, flip_immediate, OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms);
calllib('Blink_C_wrapper', 'ImageWriteComplete', board_number, timeout_ms);
%% init fresnel parameters and upload
focal_mm = 293.5;
ax_mrad = 0;
ay_mrad = 0;
phase_scale = 0.7;
wavelength = 550e-9;
pixel_size = 9.2e-6;
uploadFresnelLensPhase_mex(board_number, width, height, focal_mm, ax_mrad, ay_mrad,...
        phase_scale, wavelength, pixel_size,...
        wait_For_Trigger, flip_immediate, OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms);
%% RUN!
% ---------- 裁剪参数 ----------
cropRatio = 0.18;  % 左右各裁剪12.5%（可调整为0.10-0.15）
% ---------- 采一帧来确定尺寸 ----------
videoFrame     = snapshot(cam);
% === 左右裁剪 ===
cropW = round(size(videoFrame,2) * cropRatio);
videoFrame = videoFrame(:, cropW+1:end-cropW, :);  % 裁剪左右
videoFrameGray = im2gray(videoFrame);
frameSize      = size(videoFrameGray);   % [rows, cols]
imageCenterX   = frameSize(2)/2;
imageCenterY   = frameSize(1)/2;
% --- 双面板显示缓冲（与原图同尺寸/类型） ---
panel = cell(1,2);
panel{1} = zeros(size(videoFrame), 'like', videoFrame); % 奇数帧面板
panel{2} = zeros(size(videoFrame), 'like', videoFrame); % 偶数帧面板

% ---------- 低分辨率处理尺寸 ----------
procH = 240;  procW = procH*frameSize(2)/frameSize(1);
scaleX = procW / frameSize(2);
scaleY = procH / frameSize(1);

% ---------- 运行与显示参数 ----------
runLoop           = true;
totalFrameCount   = 0;
fpsUpdateInterval = 10;
startTime         = tic;
frameCount        = 0;
fps               = 0;

% ========== 定义“奇/偶”两套系统 ==========
% 说明：sys(1) = 奇数帧通道；sys(2) = 偶数帧通道
for k = 1:2
    % ---- 跟踪状态 ----
    sys(k).numPts          = 0;
    sys(k).oldPointsSmall  = [];
    sys(k).bboxPointsSmall = [];

    % ---- P(D) 参数（可分别调整；先与原值相同）----
    sys(k).KpX = 10;   sys(k).KdX = 0.2;
    sys(k).KpY = 5;    sys(k).KdY = 0.1;
    sys(k).tauDX = 0.08;    % 导数一阶滤波常数(s)
    sys(k).tauDY = 0.08;
    sys(k).deadband = 0.02; % 死区(相对0~1)

    % ---- 云台/输出约束
    sys(k).rampToZero = 5;                       % 无人脸时回零速度 mrad/帧
    sys(k).maxStep = 10;                         % 每帧最大步进 mrad
    sys(k).uminY = -20;  sys(k).umaxY = 20; 
    sys(k).uminX = -20;  sys(k).umaxX = 20;

    % ---- 控制状态 ----
    sys(k).dFiltX = 0;  sys(k).dFiltY = 0;
    sys(k).prevErrX = 0; sys(k).prevErrY = 0;
    sys(k).tPrev = 0;
    sys(k).slmCmdX = 0; sys(k).slmCmdY = 0;
end
% ---- 人脸检测与跟踪对象（彼此独立）----
sys(1).detector = vision.CascadeObjectDetector('stop_sign_classifier_2.xml','MergeThreshold',4,'MinSize',[25 25],'MaxSize',[50 50]); % 或者按需更换模型
sys(1).tracker  = vision.PointTracker('MaxBidirectionalError', 2);
sys(2).detector = vision.CascadeObjectDetector('MergeThreshold',4,'MinSize',[25 25],'MaxSize',[50 50]); % 或者按需更换模型
sys(2).tracker  = vision.PointTracker('MaxBidirectionalError', 2);
% 全局控制时钟
ctrlClock = tic;


% ========== 主循环 ==========
while runLoop && totalFrameCount < inf
    % 1) 选择本帧通道（奇=1，偶=2）
    % 注意：本次循环将要处理的帧序号是 totalFrameCount+1
    if mod(totalFrameCount+1, 2)==1
        idx = 1; % 奇数帧 -> sys(1)
    else
        idx = 2; % 偶数帧 -> sys(2)
    end

    % 2) 循环开始前：把云台复位到本通道的当前角度
    uploadFresnelLensPhase_mex( ...
        board_number, width, height, focal_mm, ...
        double(sys(idx).slmCmdX), double(sys(idx).slmCmdY), ...
        phase_scale, wavelength, pixel_size, ...
        wait_For_Trigger, flip_immediate, ...
        OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms);
    pause(0.1);

    % 3) 取帧
    videoFrame     = snapshot(cam);

    % === 左右裁剪 ===
    videoFrame = videoFrame(:, cropW+1:end-cropW, :);  % 裁剪左右

    videoFrameGray = im2gray(videoFrame);
    totalFrameCount = totalFrameCount + 1;
    frameCount      = frameCount + 1;

    % 生成低分辨率处理帧
    procFrameGray  = imresize(videoFrameGray, [procH, procW], 'Method', 'bilinear');

    % FPS
    if mod(frameCount, fpsUpdateInterval) == 0
        elapsedTime = toc(startTime);
        fps = frameCount / max(eps, elapsedTime);
        frameCount = 0;
        startTime  = tic;
    end

    % 初始化本帧中心/偏差
    faceCenterX = []; faceCenterY = [];
    deviationX = 0;   deviationY = 0;

    % ========= 检测/跟踪（在各自通道的状态上进行）=========
    if sys(idx).numPts < 10
        % ---- 小图做人脸检测 ----
        bboxSmall = step(sys(idx).detector, procFrameGray); % [x y w h] in small
        if ~isempty(bboxSmall)
            % 取首个框，求小图中心
            cxs = bboxSmall(1,1) + bboxSmall(1,3)/2;
            cys = bboxSmall(1,2) + bboxSmall(1,4)/2;
            % 放大回原图坐标
            faceCenterX = cxs / scaleX;
            faceCenterY = cys / scaleY;

            % 角点检测（小图ROI）
            ptsSmall = detectMinEigenFeatures(procFrameGray, 'ROI', bboxSmall(1,:));
            xySmall  = selectStrongest(ptsSmall, 60).Location;

            sys(idx).numPts = size(xySmall,1);

            % 初始化本通道tracker
            release(sys(idx).tracker);
            initialize(sys(idx).tracker, xySmall, procFrameGray);
            sys(idx).oldPointsSmall  = xySmall;
            sys(idx).bboxPointsSmall = bbox2points(bboxSmall(1,:));
        end
    else
        % ---- 小图做KLT跟踪 ----
        [xySmall, isFound] = step(sys(idx).tracker, procFrameGray);
        visibleSmall = xySmall(isFound,:);
        oldInliersSm = sys(idx).oldPointsSmall(isFound,:);
        sys(idx).numPts = size(visibleSmall,1);

        if sys(idx).numPts >= 10
            [xformSm, inlierIdx] = estgeotform2d(oldInliersSm, visibleSmall, ...
                                'similarity','MaxDistance',4);
            oldInliersSm = oldInliersSm(inlierIdx,:);
            visibleSmall = visibleSmall(inlierIdx,:);

            if ~isempty(sys(idx).bboxPointsSmall)
                sys(idx).bboxPointsSmall = transformPointsForward(xformSm, sys(idx).bboxPointsSmall);
                faceCenterX = mean(sys(idx).bboxPointsSmall(:,1)) / scaleX;
                faceCenterY = mean(sys(idx).bboxPointsSmall(:,2)) / scaleY;
            end

            % 更新点集
            sys(idx).oldPointsSmall = visibleSmall;
            setPoints(sys(idx).tracker, sys(idx).oldPointsSmall);
        else
            % 丢失：回到检测模式
            sys(idx).numPts = 0;
        end
    end

    % ========= P(D) 控制（只作用在本通道状态/参数上）=========
    if ~isempty(faceCenterX) && ~isempty(faceCenterY)
        % 时间步长（按各自通道记录）
        tNow = toc(ctrlClock);
        dt   = tNow - sys(idx).tPrev;
        sys(idx).tPrev = tNow;
        dt = max(0.001, min(dt, 0.1));  % 限定 dt

        % 归一化偏差（右/下为正）
        deviationX = (faceCenterX - imageCenterX) / (frameSize(2)/2);
        deviationY = (faceCenterY - imageCenterY) / (frameSize(1)/2);

        % 死区
        if abs(deviationX) < sys(idx).deadband, deviationX = 0; end
        if abs(deviationY) < sys(idx).deadband, deviationY = 0; end

        % 误差（保持你原符号）
        errX = -deviationX;
        errY =  deviationY;

        % P
        pTermX = sys(idx).KpX * errX;
        pTermY = sys(idx).KpY * errY;

        % D（带一阶滤波）
        if dt > 0
            dRawX = (errX - sys(idx).prevErrX) / dt;
            aX = dt / (sys(idx).tauDX + dt);
            sys(idx).dFiltX = aX * dRawX + (1 - aX) * sys(idx).dFiltX;

            dRawY = (errY - sys(idx).prevErrY) / dt;
            aY = dt / (sys(idx).tauDY + dt);
            sys(idx).dFiltY = aY * dRawY + (1 - aY) * sys(idx).dFiltY;
        end
        dTermX = sys(idx).KdX * sys(idx).dFiltX;
        dTermY = sys(idx).KdY * sys(idx).dFiltY;

        % 合成步进并限每帧最大步进
        stepX = clip(pTermX + dTermX, -sys(idx).maxStep, +sys(idx).maxStep);
        stepY = clip(pTermY + dTermY, -sys(idx).maxStep, +sys(idx).maxStep);

        % 更新该通道的 SLM 命令并限幅
        sys(idx).slmCmdX = clip(sys(idx).slmCmdX + stepX, sys(idx).uminX, sys(idx).umaxX);
        sys(idx).slmCmdY = clip(sys(idx).slmCmdY + stepY, sys(idx).uminY, sys(idx).umaxY);

        % 保存误差
        sys(idx).prevErrX = errX;  sys(idx).prevErrY = errY;

        % 可视化
        videoFrame = insertShape(videoFrame, 'Circle', [faceCenterX, faceCenterY, 20], ...
            'LineWidth', 3, 'Color', 'yellow');

        % 信息叠加
        deviationStr = sprintf('[%s] Dev X=%.1f%% Y=%.1f%%', ...
            ternary(idx==1,'Odd','Even'), deviationX*100, deviationY*100);
        videoFrame = insertText(videoFrame,[10 10], deviationStr, ...
            'FontSize',16,'BoxColor','cyan','BoxOpacity',0.6);

        slmStr = sprintf('[%s] SLM X=%.2f Y=%.2f (step %.2f/%.2f)', ...
            ternary(idx==1,'Odd','Even'), sys(idx).slmCmdX, sys(idx).slmCmdY, stepX, stepY);
        videoFrame = insertText(videoFrame,[10 40], slmStr, ...
            'FontSize',16,'BoxColor','magenta','BoxOpacity',0.6);
    else
        % 没有人脸：该通道缓慢回零（只改该通道状态）
        sys(idx).slmCmdX = sys(idx).slmCmdX - clip(sys(idx).slmCmdX, -sys(idx).rampToZero, sys(idx).rampToZero);
        sys(idx).slmCmdY = sys(idx).slmCmdY - clip(sys(idx).slmCmdY, -sys(idx).rampToZero, sys(idx).rampToZero);

        % 接近零位时可重置导数
        if abs(sys(idx).slmCmdX) < 0.5 && abs(sys(idx).slmCmdY) < 0.5
            sys(idx).dFiltX = 0;  sys(idx).dFiltY = 0;
        end
    end

    % ====== 将当前通道的叠加结果放入对应面板 ======
    panel{idx} = videoFrame;  % idx==1(奇) 或 2(偶)
    % ====== 拼接并排显示 ======
    w = size(panel{1}, 2);    % 单面板宽
    h = size(panel{1}, 1);    % 单面板高

    % 防御：若另一面板还没更新，确保尺寸一致
    if isempty(panel{3-idx})
        panel{3-idx} = zeros(size(panel{idx}), 'like', panel{idx});
    end

    composite = cat(2, panel{1}, panel{2});  % 左奇右偶
    % ====== 仅在合成图上叠加全局信息（时间/FPS）======
    globalTxt = sprintf('Time: %.2f s   FPS: %.1f', toc(ctrlClock), fps);
    % 把位置放在合成图上方中间，避免遮住左/右面板自己的信息
    posX = round(size(composite,2)/2 - 200);  % 180是个大致的偏移，方便居中
    posY = 10;
    composite = insertText(composite, [posX posY], globalTxt, ...
        'FontSize', 16, 'BoxColor', 'yellow', 'BoxOpacity', 0.6);

    % ====== 显示 ======
    step(videoPlayer, composite);

    % 窗口是否关闭
    runLoop = isOpen(videoPlayer);
end
% ===== 本地工具函数 =====
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

%% 清理
try
    clear cam;
catch, end
release(videoPlayer);

% 退出前关闭 SDK
try
    calllib('Blink_C_wrapper','Delete_SDK');
catch, end

if libisloaded('Blink_C_wrapper'), unloadlibrary('Blink_C_wrapper'); end
if libisloaded('ImageGen'),        unloadlibrary('ImageGen');        end
disp('Unconnected');