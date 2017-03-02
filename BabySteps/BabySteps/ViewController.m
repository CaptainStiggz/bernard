//
//  ViewController.m
//  BabySteps
//
//  Created by Zachary Stiggelbout on 2/16/17.
//  Copyright © 2017 Arc Reactor. All rights reserved.
//

#import "ViewController.h"
#import <AVFoundation/AVFoundation.h>

@interface ViewController ()

@property UIView *cameraPreviewView;
@property AVCaptureVideoPreviewLayer *cameraPreviewLayer;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.cameraPreviewView = [[UIView alloc] initWithFrame:self.view.frame];
    self.cameraPreviewView.backgroundColor = [UIColor blackColor];
    [self.view addSubview:self.cameraPreviewView];
    
    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPresetHigh;
    
    // add the camera
    AVCaptureDevice *camera = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceInput *video = [AVCaptureDeviceInput deviceInputWithDevice:camera error:nil];
    if (!video) {
        NSLog(@"Couldn't create video capture device");
    }
    [session addInput:video];
    
    // add the microphone
    AVCaptureDevice *microphone = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
    AVCaptureDeviceInput *audio = [AVCaptureDeviceInput deviceInputWithDevice:microphone error:nil];
    if (!audio) {
        NSLog(@"Couldn't create audio capture device");
    }
    [session addInput:audio];

    // get the video output
    
//    AVCaptureAudioDataOutput *audioDataOutput = <#Get the audio data output#>;
//    NSArray *connections = audioDataOutput.connections;
//    if ([connections count] > 0) {
//        // There should be only one connection to an AVCaptureAudioDataOutput.
//        AVCaptureConnection *connection = [connections objectAtIndex:0];
//        
//        NSArray *audioChannels = connection.audioChannels;
//        
//        for (AVCaptureAudioChannel *channel in audioChannels) {
//            float avg = channel.averagePowerLevel;
//            float peak = channel.peakHoldLevel;
//            // Update the level meter user interface.
//        }
//    }

    
    dispatch_async(dispatch_get_main_queue(), ^{
        AVCaptureVideoPreviewLayer *newCaptureVideoPreviewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];
        UIView *view = self.cameraPreviewView;
        CALayer *viewLayer = [view layer];
        
        newCaptureVideoPreviewLayer.frame = view.bounds;
        
        [viewLayer addSublayer:newCaptureVideoPreviewLayer];
        
        self.cameraPreviewLayer = newCaptureVideoPreviewLayer;
        
        [session startRunning];
    });
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
