//
//  ViewController.swift
//  BernardAV
//
//  Created by Zachary Stiggelbout on 2/17/17.
//  Copyright © 2017 Arc Reactor. All rights reserved.
//

import UIKit
import lf
import AVFoundation
import VideoToolbox

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = UIColor.black;
        self.connectRTMP(path: "rtmp://10.1.10.135/live", name: "ios");
        
    }
    
    func connectRTMP(path: String, name: String) {
        let rtmpConnection:RTMPConnection = RTMPConnection()
        let rtmpStream:RTMPStream = RTMPStream(connection: rtmpConnection)
        rtmpStream.attachAudio(AVCaptureDevice.defaultDevice(withMediaType: AVMediaTypeAudio))
        rtmpStream.attachCamera(DeviceUtil.device(withPosition: .back))
        
        // stream options
        rtmpStream.videoSettings = [
            "width": 640, // video output width
            "height": 360, // video output height
            "bitrate": 160 * 1024, // video output bitrate
            // "dataRateLimits": [160 * 1024 / 8, 1], optional kVTCompressionPropertyKey_DataRateLimits property
            "profileLevel": kVTProfileLevel_H264_Baseline_3_1, // H264 Profile require "import VideoToolbox"
            "maxKeyFrameIntervalDuration": 2, // key frame / sec
        ]
        
        // create preview view
        let lfView:LFView = LFView(frame: view.bounds)
        lfView.videoGravity = AVLayerVideoGravityResizeAspectFill
        lfView.attachStream(rtmpStream)
        self.view.addSubview(lfView)
        
        rtmpConnection.connect("\(path)");
        rtmpStream.publish(name)
        // if you want to record a stream.
        // rtmpStream.publish("streamName", type: .localRecord)
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

