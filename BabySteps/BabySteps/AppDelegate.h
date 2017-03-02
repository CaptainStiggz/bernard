//
//  AppDelegate.h
//  BabySteps
//
//  Created by Zachary Stiggelbout on 2/16/17.
//  Copyright © 2017 Arc Reactor. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <CoreData/CoreData.h>

@interface AppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;

@property (readonly, strong) NSPersistentContainer *persistentContainer;

- (void)saveContext;


@end

