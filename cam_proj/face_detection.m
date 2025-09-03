%% init camera
% % install webcam add-on
% matlab.internal.addons.showAddon('USBWEBCAM');

% Create the face detector object.
faceDetector = vision.CascadeObjectDetector('MergeThreshold',4,'MinSize',[20 20],'MaxSize',[50 50]);
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
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
phase_scale = 0.9;
wavelength = 550e-9;
pixel_size = 9.2e-6;
uploadFresnelLensPhase_mex(board_number, width, height, focal_mm, ax_mrad, ay_mrad,...
        phase_scale, wavelength, pixel_size,...
        wait_For_Trigger, flip_immediate, OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms);
%% RUN!
% ---------- 采一帧来确定尺寸 ----------
videoFrame    = snapshot(cam);
videoFrameGray= im2gray(videoFrame);
frameSize     = size(videoFrameGray);   % [rows, cols]
imageCenterX  = frameSize(2)/2;
imageCenterY  = frameSize(1)/2;

% ---------- 低分辨率处理尺寸（按需改成 [480 640] 或 [240 320]） ----------
procH = 240;               % 处理用高度
procW = 320;               % 处理用宽度
scaleX = procW / frameSize(2);
scaleY = procH / frameSize(1);

% ---------- 运行与显示参数 ----------
runLoop            = true;
numPts             = 0;
totalFrameCount    = 0;
fpsUpdateInterval  = 10;
startTime          = tic;
frameCount         = 0;
fps                = 0;

% ---------- PID 参数（可调） ----------
KpX = 13;  KiX = 0;  KdX = 0.32;
KpY = 5;  KiY = 0;  KdY = 0.12;
tauDX = 0.08;  % X轴导数一阶滤波时间常数（秒）
tauDY = 0.08;  % Y轴导数一阶滤波时间常数（秒）
KawX  = 0;   % X轴 anti-windup 回算增益
KawY  = 0;   % Y轴 anti-windup 回算增益

deadband = 0.01;        % 2% 死区
umin = -25; umax =  25; % 输出角度限幅（mrad）
maxStep_mrad_per_frame = 15; % 每帧最大步进，抑制跳变

% ---------- PID 状态 ----------
intX = 0; intY = 0;           % 积分项
dFiltX = 0; dFiltY = 0;       % 滤波后的导数
prevErrX = 0; prevErrY = 0;   % 上一帧误差（用于导数）
ctrlClock = tic; tPrev = 0;   % 控制用时钟

% ---------- SLM状态 ----------
slmCmdX = 0; slmCmdY = 0;
prevSlmCmdX = 0; prevSlmCmdY = 0;  % 用于anti-windup

% ---------- 跟踪框变量 ----------
bboxPoints = []; oldPoints = [];

% ========== 主循环 ==========
while runLoop && totalFrameCount < inf
    % 获取帧
    videoFrame     = snapshot(cam);
    videoFrameGray = im2gray(videoFrame);
    totalFrameCount = totalFrameCount + 1;
    frameCount      = frameCount + 1;
    % == 生成低分辨率处理帧 ==
    procFrameGray  = imresize(videoFrameGray, [procH, procW], 'Method', 'bilinear');

    % FPS
    if mod(frameCount, fpsUpdateInterval) == 0
        elapsedTime = toc(startTime);
        fps = frameCount / max(eps, elapsedTime);
        frameCount = 0;
        startTime  = tic;
    end

    % 初始化本帧的人脸中心/偏差
    faceCenterX = []; faceCenterY = [];
    deviationX = 0;   deviationY = 0;

    % 检测/跟踪
    if numPts < 10
        % ---- 小图上做人脸检测 ----
        bboxSmall = step(faceDetector, procFrameGray); % [x y w h] in small
        if ~isempty(bboxSmall)
            % 取首个框，先在小图求中心
            faceCenterX_small = bboxSmall(1,1) + bboxSmall(1,3)/2;
            faceCenterY_small = bboxSmall(1,2) + bboxSmall(1,4)/2;
            % 放大回原图
            faceCenterX = faceCenterX_small / scaleX;
            faceCenterY = faceCenterY_small / scaleY;
    
            % 小图上检测角点
            ptsSmall = detectMinEigenFeatures(procFrameGray, 'ROI', bboxSmall(1,:));
            % 限制点数可减负担（可选）
            % xySmall  = ptsSmall.Location;
            xySmall  = selectStrongest(ptsSmall, 60).Location;
    
            numPts  = size(xySmall,1);
    
            % 初始化“小图 tracker”
            release(pointTracker);
            initialize(pointTracker, xySmall, procFrameGray);
            oldPointsSmall = xySmall;
    
            % 框四角（小图）
            bboxPointsSmall = bbox2points(bboxSmall(1,:)); % 4x2
            % % 放大到原图用于显示
            % bboxPoints = [bboxPointsSmall(:,1)/scaleX, bboxPointsSmall(:,2)/scaleY];
            % % 画框（在原图上）
            % bboxPolygon = reshape(bboxPoints',1,[]);
            % videoFrame  = insertShape(videoFrame,'Polygon',bboxPolygon,'LineWidth',3);
        end
    else
        % ---- 小图上做KLT跟踪 ----
        [xySmall, isFound] = step(pointTracker, procFrameGray);
        visibleSmall = xySmall(isFound,:);
        oldInliersSm = oldPointsSmall(isFound,:);
        numPts       = size(visibleSmall,1);
    
        if numPts >= 10
            % 在小图坐标系估计相似变换
            [xformSm, inlierIdx] = estgeotform2d(oldInliersSm, visibleSmall, ...
                                                 'similarity','MaxDistance',4);
            oldInliersSm = oldInliersSm(inlierIdx,:);
            visibleSmall = visibleSmall(inlierIdx,:);
    
            % 更新小图框四角
            if exist('bboxPointsSmall','var') && ~isempty(bboxPointsSmall)
                bboxPointsSmall = transformPointsForward(xformSm, bboxPointsSmall);
                % 小图中心 -> 原图中心
                faceCenterX = mean(bboxPointsSmall(:,1)) / scaleX;
                faceCenterY = mean(bboxPointsSmall(:,2)) / scaleY;
    
                % % 放大后在原图画框
                % bboxPoints = [bboxPointsSmall(:,1)/scaleX, bboxPointsSmall(:,2)/scaleY];
                % bboxPolygon = reshape(bboxPoints',1,[]);
                % videoFrame  = insertShape(videoFrame,'Polygon',bboxPolygon,'LineWidth',3);
            end
    
            % 重置点（小图坐标）
            oldPointsSmall = visibleSmall;
            setPoints(pointTracker, oldPointsSmall);
        else
            % 点丢失：回到检测模式
            numPts = 0;
        end
    end

    % PID 控制（每帧最多一次下发）
    if ~isempty(faceCenterX) && ~isempty(faceCenterY)
        % 获取时间增量
        tNow = toc(ctrlClock);
        dt = tNow - tPrev;
        tPrev = tNow;
        % 防止dt过小或过大
        dt = max(0.001, min(dt, 0.1));
        % 归一化偏差（右/下为正）
        deviationX = (faceCenterX - imageCenterX) / (frameSize(2)/2);
        deviationY = (faceCenterY - imageCenterY) / (frameSize(1)/2);

        % 死区
        if abs(deviationX) < deadband, deviationX = 0; end
        if abs(deviationY) < deadband, deviationY = 0; end

        % 误差定义（保持你原先几何符号）
        errX = -deviationX;
        errY =  deviationY;

        % P项
        pTermX = KpX * errX;
        pTermY = KpY * errY;
        
        % I项 (带anti-windup)
        intX = intX + KiX * errX * dt;
        intY = intY + KiY * errY * dt;
        
        % D项 (带一阶滤波)
        if dt > 0
            dRawX = (errX - prevErrX) / dt;
            alphaX = dt / (tauDX + dt);  % 一阶滤波系数
            dFiltX = alphaX * dRawX + (1 - alphaX) * dFiltX;
        end
        dTermX = KdX * dFiltX;
        if dt > 0
            dRawY = (errY - prevErrY) / dt;
            alphaY = dt / (tauDY + dt);  % 一阶滤波系数
            dFiltY = alphaY * dRawY + (1 - alphaY) * dFiltY;
        end
        dTermY = KdY * dFiltY;
        % 组合PID输出
        stepX = pTermX + intX + dTermX;
        stepY = pTermY + intY + dTermY;
        % 每帧最大步进（柔化命令）
        stepX = clip(stepX, -maxStep_mrad_per_frame, +maxStep_mrad_per_frame);
        stepY = clip(stepY, -maxStep_mrad_per_frame, +maxStep_mrad_per_frame);
        % 保存限幅前的命令
        prevSlmCmdX = slmCmdX;
        prevSlmCmdY = slmCmdY;
        % SLM 步进
        slmCmdX = slmCmdX + stepX;
        slmCmdY = slmCmdY + stepY;
        % SLM当前位置限幅
        slmCmdX = clip(slmCmdX, umin, umax);
        slmCmdY = clip(slmCmdY, umin, umax);
        
        % Anti-windup: 如果输出饱和，回算积分项
        if slmCmdX ~= prevSlmCmdX + stepX  % X轴饱和
            intX = intX - KawX * (slmCmdX - (prevSlmCmdX + stepX)) * dt;
        end
        if slmCmdY ~= prevSlmCmdY + stepY  % Y轴饱和
            intY = intY - KawY * (slmCmdY - (prevSlmCmdY + stepY)) * dt;
        end
        
        % 限制积分项防止过度累积
        intMaxX = maxStep_mrad_per_frame * 2;  % 可调整
        intMaxY = maxStep_mrad_per_frame * 2;
        intX = clip(intX, -intMaxX, intMaxX);
        intY = clip(intY, -intMaxY, intMaxY);

        % 更新上一帧误差
        prevErrX = errX;
        prevErrY = errY;

        % === 下发相位（每帧一次） ===
        uploadFresnelLensPhase_mex(board_number, width, height, focal_mm, ...
            double(slmCmdX), double(slmCmdY), ...
            phase_scale, wavelength, pixel_size, ...
            wait_For_Trigger, flip_immediate, ...
            OutputPulseImageFlip, OutputPulseImageRefresh, timeout_ms);
        % 画人脸中心
        videoFrame = insertShape(videoFrame, 'Circle', ...
            [faceCenterX, faceCenterY, 20], ...  % [x, y, radius]
            'LineWidth', 3, ...  % 调整线条粗细
            'Color', 'yellow');
        pause(0.03)
    else
        % 未检测到人脸 缓慢回零
        rampToZero = 2; % mrad/帧
        slmCmdX = slmCmdX - clip(slmCmdX, -rampToZero, rampToZero);
        slmCmdY = slmCmdY - clip(slmCmdY, -rampToZero, rampToZero);
        
        % 重置积分项和导数滤波（可选）
        if abs(slmCmdX) < 0.5 && abs(slmCmdY) < 0.5  % 接近零位时重置
            intX = 0; intY = 0;
            dFiltX = 0; dFiltY = 0;
        end
    end

    % 文本

    if ~isempty(faceCenterX) && ~isempty(faceCenterY)
        deviationStr = sprintf('Deviation: X=%.1f%%, Y=%.1f%%', deviationX*100, deviationY*100);
        videoFrame = insertText(videoFrame,[10 10], deviationStr, ...
            'FontSize',16,'BoxColor','cyan','BoxOpacity',0.6);

        slmStr = sprintf('SLM Angle: X=%.2f mrad, Y=%.2f mrad', slmCmdX, slmCmdY);
        videoFrame = insertText(videoFrame,[10 40], slmStr, ...
            'FontSize',16,'BoxColor','magenta','BoxOpacity',0.6);

        slmStr = sprintf('SLM Step: X=%.2f mrad, Y=%.2f mrad', stepX, stepY);
        videoFrame = insertText(videoFrame,[10 70], slmStr, ...
            'FontSize',16,'BoxColor','green','BoxOpacity',0.6);
    end

    % textStr = sprintf('FPS: %.1f', fps);
    % videoFrame = insertText(videoFrame,[10 100], textStr, ...
    %     'FontSize',18,'BoxColor','yellow','BoxOpacity',0.8);

    % 在视频帧上插入时间戳文本
    textStr = sprintf('Time: %.2f s', toc(ctrlClock));
    videoFrame = insertText(videoFrame, [10 100], textStr, ...
    'FontSize', 16, 'BoxColor', 'yellow', 'BoxOpacity', 0.6);

    % 显示
    step(videoPlayer, videoFrame);

    % 窗口是否关闭
    runLoop = isOpen(videoPlayer);
end

% 清理
try
    clear cam;
catch, end
release(videoPlayer);
release(pointTracker);
release(faceDetector);

% 退出前关闭 SDK
try
    calllib('Blink_C_wrapper','Delete_SDK');
catch, end

if libisloaded('Blink_C_wrapper'), unloadlibrary('Blink_C_wrapper'); end
if libisloaded('ImageGen'),        unloadlibrary('ImageGen');        end
disp('Unconnected');