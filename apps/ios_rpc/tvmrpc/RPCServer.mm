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

/*!
 * \file ViewController.mm
 */

#import "RPCServer.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

// TODO: will be rewrite
#include "rpc_server.h"

#include <string>

constexpr int kRPCMagic = 0xff271;

namespace tvm {
namespace runtime {

/*!
 * \brief Message handling function for event driven server.
 *
 * \param in_bytes The incoming bytes.
 * \param event_flag  1: read_available, 2: write_avaiable.
 * \return State flag.
 *     1: continue running, no need to write,
 *     2: need to write
 *     0: shutdown
 */
using FEventHandler = PackedFunc;

/*!
 * \brief Create a server event handler.
 *
 * \param outputStream The output stream used to send outputs.
 * \param name The name of the server.
 * \param remote_key The remote key
 * \return The event handler.
 */
FEventHandler CreateServerEventHandler(NSOutputStream* outputStream, std::string name,
                                       std::string remote_key) {
  const PackedFunc* event_handler_factor = Registry::Get("rpc.CreateEventDrivenServer");
  ICHECK(event_handler_factor != nullptr)
    << "You are using tvm_runtime module built without RPC support. "
    << "Please rebuild it with USE_RPC flag.";

  PackedFunc writer_func([outputStream](TVMArgs args, TVMRetValue* rv) {
    TVMByteArray *data = args[0].ptr<TVMByteArray>();
    int64_t nbytes = [outputStream write:reinterpret_cast<const uint8_t*>(data->data) maxLength:data->size];
    if (nbytes < 0) {
      NSLog(@"%@", [outputStream streamError].localizedDescription);
      throw tvm::Error("Stream error");
    }
    *rv = nbytes;
  });

  return (*event_handler_factor)(writer_func, name, remote_key);
}

}  // namespace runtime
}  // namespace tvm

@implementation RPCServer {
  // Connection mode of RPC server
  RPCServerMode mode_;
  // Event listener
  id <RPCServerEvenlListener> delegate_;
  // Worker thread
  NSThread* worker_thread_;
  // Triger to continue processing
  BOOL shouldKeepRunning;
  // Input socket stream
  NSInputStream* inputStream_;
  // Output socket stream
  NSOutputStream* outputStream_;
  // Temporal receive buffer.
  std::string recvBuffer_;
  // Whether connection is initialized.
  bool initialized_;
  // The key of the server.
  NSString* key_;
  // The url of host.
  NSString* url_;
  // The port of host.
  NSInteger port_;
  // Initial bytes to be send to remote
  std::string initBytes_;
  // Send pointer of initial bytes.
  size_t initSendPtr_;
  // Event handler.
  tvm::runtime::FEventHandler handler_;
}


- (instancetype)init {
  [super init];
  return self;
}

- (instancetype)initWithMode:(RPCServerMode) mode {
  [super init];
  mode_ = mode;
  return self;
}

- (void)setDelegate:(id<RPCServerEvenlListener>) delegate {
  delegate_ = delegate;
}

- (void)stream:(NSStream*)strm handleEvent:(NSStreamEvent)event {
  std::string buffer;
  switch (event) {
    case NSStreamEventOpenCompleted: {
      [self notifyState:TrackerConnected];
      break;
    }
    case NSStreamEventHasBytesAvailable:
      if (strm == inputStream_) {
        [self onReadAvailable];
      }
      break;
    case NSStreamEventHasSpaceAvailable: {
      if (strm == outputStream_) {
        [self onWriteAvailable];
      }
      break;
    }
    case NSStreamEventErrorOccurred: {
      NSLog(@"%@", [strm streamError].localizedDescription);
      break;
    }
    case NSStreamEventEndEncountered: {
      [self close];
      // auto reconnect when normal end.
      [self open];
      break;
    }
    default: {
      NSLog(@"Unknown event");
    }
  }
}

-(void)notifyError:(NSString*) msg {
  NSLog(@"[IOS-RPC] ERROR: %@", msg);
  if (delegate_ != nil)
    [delegate_ onError:msg];
}

-(void)notifyState:(RPCServerStatus) state {
  // print status to output to notify host runner script about rpc server status
  NSLog(@"[IOS-RPC] STATE: %d", state);
  if (delegate_ != nil)
    [delegate_ onStatusChanged:state];
}

- (void)onReadAvailable {
  if (!initialized_) {
    int code;
    size_t nbytes = [inputStream_ read:reinterpret_cast<uint8_t*>(&code) maxLength:sizeof(code)];
    if (nbytes != sizeof(code)) {
      [self notifyError:@"Fail to receive remote confirmation code."];
      [self close];
    } else if (code == kRPCMagic + 2) {
      [self notifyError:@"Proxy server cannot find client that matches the key"];
      [self close];
    } else if (code == kRPCMagic + 1) {
      [self notifyError:@"Proxy server already have another server with same key"];
      [self close];
    } else if (code != kRPCMagic) {
      [self notifyError:@"Given address is not a TVM RPC Proxy"];
      [self close];
    } else {
      initialized_ = true;
      [self notifyState:RPCSessionStarted];
      ICHECK(handler_ != nullptr);
    }
  }
  const int kBufferSize = 4 << 10;
  if (initialized_) {
    while ([inputStream_ hasBytesAvailable]) {
      recvBuffer_.resize(kBufferSize);
      uint8_t* bptr = reinterpret_cast<uint8_t*>(&recvBuffer_[0]);
      size_t nbytes = [inputStream_ read:bptr maxLength:kBufferSize];
      recvBuffer_.resize(nbytes);
      int flag = 1;
      if ([outputStream_ hasSpaceAvailable]) {
        flag |= 2;
      }
      // always try to write
      try {
        TVMByteArray arr {recvBuffer_.data(), recvBuffer_.size()};
        flag = handler_(arr, flag);
        if (flag == 2) {
          [self onShutdownReceived];
        }
      } catch (const tvm::Error& e) {
        [self close];
      }
    }
  }
}

- (void)onShutdownReceived {
  [self close];
}

- (void)onWriteAvailable {
  if (initSendPtr_ < initBytes_.length()) {
    initSendPtr_ += [outputStream_ write:reinterpret_cast<uint8_t*>(&initBytes_[initSendPtr_])
                               maxLength:(initBytes_.length() - initSendPtr_)];
  }
  if (initialized_) {
    try {
      TVMByteArray dummy {nullptr, 0};
      int flag = handler_(dummy, 2);
      if (flag == 2) {
        [self onShutdownReceived];
      }
    } catch (const tvm::Error& e) {
      [self close];
    }
  }
}

- (void)open {
  // Initialize the data states.
  std::string full_key = std::string("server:") + [key_ UTF8String];
  std::ostringstream os;
  int rpc_magic = kRPCMagic;
  os.write(reinterpret_cast<char*>(&rpc_magic), sizeof(rpc_magic));
  int keylen = static_cast<int>(full_key.length());
  os.write(reinterpret_cast<char*>(&keylen), sizeof(keylen));
  os.write(full_key.c_str(), full_key.length());
  initialized_ = false;
  initBytes_ = os.str();
  initSendPtr_ = 0;
  
  // Initialize the network.
  CFReadStreamRef readStream;
  CFWriteStreamRef writeStream;
  if (mode_ == RPCServerMode_Proxy) {
    CFStreamCreatePairWithSocketToHost(NULL, (CFStringRef)url_, port_,
                                       &readStream, &writeStream);
  } else { // RPCServerMode_Pure
    CFSocketNativeHandle socket = 0;
    CFStreamCreatePairWithSocket(NULL, socket, &readStream, &writeStream);
  }
  inputStream_ = (NSInputStream*)readStream;
  outputStream_ = (NSOutputStream*)writeStream;
  [inputStream_ setDelegate:self];
  [outputStream_ setDelegate:self];
  [inputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ scheduleInRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ open];
  [inputStream_ open];

  handler_ = tvm::runtime::CreateServerEventHandler(outputStream_, full_key, "%toinit");
  ICHECK(handler_ != nullptr);
}

- (void)close {
  NSLog(@"Closing the streams.");
  [inputStream_ close];
  [outputStream_ close];
  [inputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [outputStream_ removeFromRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
  [inputStream_ setDelegate:nil];
  [outputStream_ setDelegate:nil];
  inputStream_ = nil;
  outputStream_ = nil;
  handler_ = nullptr;
}

static void handleConnect(CFSocketRef socket, CFSocketCallBackType type, CFDataRef address, const void *data, void *info) {

};

- (void)startWithHost:(NSString*) host port: (int) port key:(NSString*) key {
  key_ = [key copy];
  port_ = port;
  url_ = [host copy];
  
  if (mode_ == RPCServerMode_Proxy) {
    // process in separate thead with runloop
    worker_thread_ = [[NSThread alloc] initWithBlock:^{
      @autoreleasepool {
        shouldKeepRunning = YES;
        [self open];
        while (shouldKeepRunning && [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate distantFuture]]);
      }
    }];
  } else if (mode_ == RPCServerMode_Tracker) {
    worker_thread_ = [[NSThread alloc] initWithBlock:^{
      @autoreleasepool {
        // part 1
        CFSocketRef myipv4cfsock = CFSocketCreate(
            kCFAllocatorDefault,
            PF_INET,
            SOCK_STREAM,
            IPPROTO_TCP,
            kCFSocketAcceptCallBack, handleConnect, NULL);

        // part 2
        struct sockaddr_in sin;

        memset(&sin, 0, sizeof(sin));
        sin.sin_len = sizeof(sin);
        sin.sin_family = AF_INET; /* Address family */
        sin.sin_port = htons(9090); /* Or a specific port */
        sin.sin_addr.s_addr= INADDR_ANY;

        CFDataRef sincfd = CFDataCreate(
            kCFAllocatorDefault,
            (UInt8 *)&sin,
            sizeof(sin));

        CFSocketSetAddress(myipv4cfsock, sincfd);
        CFRelease(sincfd);

        // part 3
        CFRunLoopSourceRef socketsource = CFSocketCreateRunLoopSource(
            kCFAllocatorDefault,
            myipv4cfsock,
            0);

        CFRunLoopAddSource(
            CFRunLoopGetCurrent(),
            socketsource,
            kCFRunLoopDefaultMode);

        shouldKeepRunning = YES;
        [self open];
        while (shouldKeepRunning && [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate distantFuture]]);
      }
    }];
  } else if (mode_ == RPCServerMode_Tracker) {
    // TODO: tracker mode are realized in different manner via sync socket processing
    // It has next limitation:
    //   - disconnect/stop interface is not implemented
    //   - do not provide info about state changes
    worker_thread_ = [[NSThread alloc] initWithBlock:^{
      tvm::runtime::RPCServer server("0.0.0.0", 9090, 9099,
                                     "('" + std::string(url_.UTF8String) + "', " + std::to_string(port_) + ")",
                                     key_.UTF8String, "");
      [self notifyState:TrackerConnected];  // WA. while real server cnnot report that
      server.Start();
    }];
  }
  [worker_thread_ start];
}

- (void)stop {
  if (worker_thread_ == nil)
    return;

  [self performSelector:@selector(stop_) onThread:worker_thread_ withObject:nil waitUntilDone:NO];
  worker_thread_ = nil;
}

- (void)stop_ {
  [self close];
  shouldKeepRunning = NO;
}

@end
