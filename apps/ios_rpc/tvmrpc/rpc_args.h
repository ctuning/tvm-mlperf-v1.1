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

#ifndef TVM_APPS_IOS_RPC_ARGS_H_
#define TVM_APPS_IOS_RPC_ARGS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RPCArgs_t {
  const char* tracker_url;
  int tracker_port;
  
  const char* key;
  const char* custom_addr;
  int port;
  int port_end;
  
  char immediate_connect;
  char proxy_mode;
} RPCArgs;

RPCArgs get_current_rpc_args(void);
void update_rpc_args(int argc, char * argv[]);
void set_current_rpc_args(RPCArgs args);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_APPS_IOS_RPC_ARGS_H_
