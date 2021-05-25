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

#import "rpc_args.h"

#import <Foundation/Foundation.h>

#import "../../../src/support/utils.h"
#import "../../../src/support/socket.h"

#import <string>

using std::string;

const char* kUsage = "\n"
"iOS tvmrpc application supported flags:\n"
"--tracker_url  - The tracker/proxy address, Default=0.0.0.0\n"
"--tracker_port - The tracker/proxy port, Default=9190\n"
"--port         - The port of the RPC, Default=9090\n"
"--port_end     - The end search port of the RPC, Default=9099\n"
"--key          - The key used to identify the device type in tracker. Default=\"\"\n"
"--custom_addr  - Custom IP Address to Report to RPC Tracker. Default=\"\"\n"
"--immediate_connect - No UI interconnection, connect to tracker immediately. Default=False\n"
"--proxy_mode   - Connect to server like a proxy instead of tracker mode. Default=False\n"
"\n";

struct RPCArgs_cpp {
  string tracker_url = "0.0.0.0";
  int tracker_port = 9190;
  
  string key;
  string custom_addr = "";
  int port = 9090;
  int port_end = 9099;
  
  bool immediate_connect = false;
  bool proxy_mode = false;
  
  operator RPCArgs() const {
    return RPCArgs {
      .tracker_url = tracker_url.c_str(),
      .tracker_port = tracker_port,
      .key = key.c_str(),
      .custom_addr = custom_addr.c_str(),
      .port = port,
      .port_end = port_end,
      .immediate_connect = immediate_connect,
      .proxy_mode = proxy_mode
    };
  };
  
  RPCArgs_cpp& operator=(const RPCArgs& args) {
    tracker_url = args.tracker_url;
    tracker_port = args.tracker_port;
    key = args.key;
    custom_addr = args.custom_addr;
    port = args.port;
    port_end = args.port_end;
    immediate_connect = args.immediate_connect;
    proxy_mode = args.proxy_mode;
    return *this;
  }
};

struct RPCArgs_cpp g_rpc_args;

static void restore_from_cache() {
  NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
  
  auto get_string_from_cache = [defaults] (const char* key) {
    NSString* ns_key = [NSString stringWithUTF8String:key];
    NSString* ns_val = [defaults stringForKey:ns_key];
    return std::string(ns_val != nil ? [ns_val UTF8String] : "");
  };
  
  auto get_int_from_cache = [defaults] (const char* key) {
    NSString* ns_key = [NSString stringWithUTF8String:key];
    return static_cast<int>([defaults integerForKey:ns_key]);
  };
  
  g_rpc_args.tracker_url = get_string_from_cache("tmvrpc_url");
  g_rpc_args.tracker_port = get_int_from_cache("tmvrpc_port");
  g_rpc_args.key = get_string_from_cache("tmvrpc_key");
}

static void update_in_cache() {
  NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
  
  [defaults setObject:[NSString stringWithUTF8String:g_rpc_args.tracker_url.c_str()] forKey:@"tmvrpc_url"];
  [defaults setInteger:g_rpc_args.tracker_port forKey:@"tmvrpc_port"];
  [defaults setObject:[NSString stringWithUTF8String:g_rpc_args.key.c_str()] forKey:@"tmvrpc_key"];
}

string GetCmdOption(int argc, char* argv[], string option, bool key = false) {
  string cmd;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg.find(option) == 0) {
      if (key) {
        cmd = argv[i];
        return cmd;
      }
      // We assume "=" is the end of option.
      ICHECK_EQ(*option.rbegin(), '=');
      cmd = arg.substr(arg.find('=') + 1);
      return cmd;
    }
  }
  return cmd;
}

void update_rpc_args(int argc, char* argv[]) {
  restore_from_cache();
  RPCArgs_cpp &args = g_rpc_args;
  
  using tvm::support::IsNumber;
  using tvm::support::ValidateIP;
  constexpr int MAX_PORT_NUM = 65535;
  
  const string immediate_connect = GetCmdOption(argc, argv, "--immediate_connect", true);
  if (!immediate_connect.empty()) {
    args.immediate_connect = true;
  }

  const string proxy_mode = GetCmdOption(argc, argv, "--proxy_mode", true);
  if (!proxy_mode.empty()) {
    args.proxy_mode = true;
  }

  const string tracker_url = GetCmdOption(argc, argv, "--tracker_url=");
  if (!tracker_url.empty()) {
    if (!ValidateIP(tracker_url)) {
      LOG(WARNING) << "Wrong tracker address format.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.tracker_url = tracker_url;
  }

  const string tracker_port = GetCmdOption(argc, argv, "--tracker_port=");
  if (!tracker_port.empty()) {
    if (!IsNumber(tracker_port) || stoi(tracker_port) > MAX_PORT_NUM) {
      LOG(WARNING) << "Wrong trackerport number.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.tracker_port = stoi(tracker_port);
  }

  const string port = GetCmdOption(argc, argv, "--port=");
  if (!port.empty()) {
    if (!IsNumber(port) || stoi(port) > MAX_PORT_NUM) {
      LOG(WARNING) << "Wrong port number.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.port = stoi(port);
  }

  const string port_end = GetCmdOption(argc, argv, "--port_end=");
  if (!port_end.empty()) {
    if (!IsNumber(port_end) || stoi(port_end) > MAX_PORT_NUM) {
      LOG(WARNING) << "Wrong port_end number.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.port_end = stoi(port_end);
  }

  const string key = GetCmdOption(argc, argv, "--key=");
  if (!key.empty()) {
    args.key = key;
  }

  const string custom_addr = GetCmdOption(argc, argv, "--custom_addr=");
  if (!custom_addr.empty()) {
    if (!ValidateIP(custom_addr)) {
      LOG(WARNING) << "Wrong custom address format.";
      LOG(INFO) << kUsage;
      exit(1);
    }
    args.custom_addr = '"' + custom_addr + '"';
  }
  
  update_in_cache();
}

RPCArgs get_current_rpc_args(void) {
  return g_rpc_args;
}

void set_current_rpc_args(RPCArgs args) {
  g_rpc_args = args;
  update_in_cache();
}
