/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#import <Foundation/Foundation.h>

typedef enum {
  TrackerConnected,
  TrackerDisconnected,
  ProxyConnected,
  ProxyDisconnected,
  RPCSessionStarted,
  RPCSessionFiniched
} RPCServerStatus;

typedef enum {
  RPCServerMode_Pure,
  RPCServerMode_Proxy,
  RPCServerMode_Tracker
} RPCServerMode;

@protocol RPCServerEvenlListener <NSObject>
- (void)onError:(NSString*) msg;
- (void)onStatusChanged:(RPCServerStatus) status;
@end

@interface RPCServer : NSObject <NSStreamDelegate>
- (instancetype)initWithMode:(RPCServerMode) mode;
- (void)setDelegate:(id<RPCServerEvenlListener>) delegate;
- (void)startWithHost:(NSString*) host port: (int) port key:(NSString*) key;
- (void)stop;
@end
