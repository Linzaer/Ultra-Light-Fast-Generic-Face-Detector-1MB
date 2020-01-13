package com.facesdk;

public class FaceSDKNative {
    //SDK初始化
    public native boolean FaceDetectionModelInit(String faceDetectionModelPath);

    //SDK人脸检测接口
    public native int[] FaceDetect(byte[] imageDate, int imageWidth , int imageHeight, int imageChannel);

    //SDK销毁
    public native boolean FaceDetectionModelUnInit();

    static {
        System.loadLibrary("facedetect");
    }

}
