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
 * \file rpc_server.h
 * \brief RPC Server implementation.
 */
#ifndef TVM_APPS_IOS_RPC_SERVER_H_
#define TVM_APPS_IOS_RPC_SERVER_H_

#include <string>
#include <future>
#include <chrono>
#include <random>
#include <dirent.h>

#include "tvm/runtime/c_runtime_api.h"
#include "../../../src/runtime/rpc/rpc_endpoint.h"
#include "../../../src/runtime/rpc/rpc_socket_impl.h"
#include "../../../src/support/socket.h"


namespace tvm {
namespace runtime {

/*!
 * \brief TrackerClient Tracker client class
 */
class TrackerClient {
 public:
  /*!
   * \brief Constructor of TrackerClient Tracker client class.
   * \param tracker_addr The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
   * \param key The key used to identify the device type in tracker. Default=""
   * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
   */
  TrackerClient(const std::string& tracker_addr, const std::string& key,
                const std::string& custom_addr)
      : tracker_addr_(tracker_addr),
        key_(key),
        custom_addr_(custom_addr),
        gen_(std::random_device{}()),
        dis_(0.0, 1.0) {}
  
  /*!
   * \brief Destructor.
   */
  ~TrackerClient() {
    // Free the resources
    Close();
  }
  
  /*!
   * \brief IsValid Check tracker is valid.
   */
  bool IsValid() { return (!tracker_addr_.empty() && !tracker_sock_.IsClosed()); }
  
  /*!
   * \brief TryConnect Connect to tracker if the tracker address is valid.
   */
  void TryConnect() {
    if (!tracker_addr_.empty() && (tracker_sock_.IsClosed())) {
      tracker_sock_ = ConnectWithRetry();

      int code = kRPCTrackerMagic;
      ICHECK_EQ(tracker_sock_.SendAll(&code, sizeof(code)), sizeof(code));
      ICHECK_EQ(tracker_sock_.RecvAll(&code, sizeof(code)), sizeof(code));
      ICHECK_EQ(code, kRPCTrackerMagic) << tracker_addr_.c_str() << " is not RPC Tracker";

      std::ostringstream ss;
      ss << "[" << static_cast<int>(TrackerCode::kUpdateInfo) << ", {\"key\": \"server:" << key_
         << "\"}]";
      tracker_sock_.SendBytes(ss.str());

      // Receive status and validate
      std::string remote_status = tracker_sock_.RecvBytes();
      ICHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
    }
  }

  /*!
   * \brief Close Clean up tracker resources.
   */
  void Close() {
    // close tracker resource
    if (!tracker_sock_.IsClosed()) {
      tracker_sock_.Close();
    }
  }
  /*!
   * \brief ReportResourceAndGetKey Report resource to tracker.
   * \param port listening port.
   * \param matchkey Random match key output.
   */
  void ReportResourceAndGetKey(int port, std::string* matchkey) {
    if (!tracker_sock_.IsClosed()) {
      *matchkey = RandomKey(key_ + ":", old_keyset_);
      if (custom_addr_.empty()) {
        custom_addr_ = "null";
      }

      std::ostringstream ss;
      ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << key_ << "\", [" << port
         << ", \"" << *matchkey << "\"], " << custom_addr_ << "]";

      tracker_sock_.SendBytes(ss.str());

      // Receive status and validate
      std::string remote_status = tracker_sock_.RecvBytes();
      ICHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
    } else {
      *matchkey = key_;
    }
  }

  /*!
   * \brief Report resource to tracker.
   * \param listen_sock Listen socket details for select.
   * \param port listening port.
   * \param ping_period Select wait time.
   * \param matchkey Random match key output.
   */
  void WaitConnectionAndUpdateKey(support::TCPSocket listen_sock, int port, int ping_period,
                                  std::string* matchkey) {
    int unmatch_period_count = 0;
    int unmatch_timeout = 4;
    while (true) {
      if (!tracker_sock_.IsClosed()) {
        support::PollHelper poller;
        poller.WatchRead(listen_sock.sockfd);
        poller.Poll(ping_period * 1000);
        if (!poller.CheckRead(listen_sock.sockfd)) {
          std::ostringstream ss;
          ss << "[" << int(TrackerCode::kGetPendingMatchKeys) << "]";
          tracker_sock_.SendBytes(ss.str());

          // Receive status and validate
          std::string pending_keys = tracker_sock_.RecvBytes();
          old_keyset_.insert(*matchkey);

          // if match key not in pending key set
          // it means the key is acquired by a client but not used.
          if (pending_keys.find(*matchkey) == std::string::npos) {
            unmatch_period_count += 1;
          } else {
            unmatch_period_count = 0;
          }
          // regenerate match key if key is acquired but not used for a while
          if (unmatch_period_count * ping_period > unmatch_timeout + ping_period) {
            LOG(INFO) << "no incoming connections, regenerate key ...";

            *matchkey = RandomKey(key_ + ":", old_keyset_);

            std::ostringstream ss;
            ss << "[" << static_cast<int>(TrackerCode::kPut) << ", \"" << key_ << "\", [" << port
               << ", \"" << *matchkey << "\"], " << custom_addr_ << "]";
            tracker_sock_.SendBytes(ss.str());

            std::string remote_status = tracker_sock_.RecvBytes();
            ICHECK_EQ(std::stoi(remote_status), static_cast<int>(TrackerCode::kSuccess));
            unmatch_period_count = 0;
          }
          continue;
        }
      }
      break;
    }
  }

 private:
  /*!
   * \brief Connect to a RPC address with retry.
            This function is only reliable to short period of server restart.
   * \param timeout Timeout during retry
   * \param retry_period Number of seconds before we retry again.
   * \return TCPSocket The socket information if connect is success.
   */
  support::TCPSocket ConnectWithRetry(int timeout = 60, int retry_period = 5) {
    auto tbegin = std::chrono::system_clock::now();
    while (true) {
      support::SockAddr addr(tracker_addr_);
      support::TCPSocket sock;
      sock.Create();
      LOG(INFO) << "Tracker connecting to " << addr.AsString();
      if (sock.Connect(addr)) {
        return sock;
      }

      auto period = (std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now() - tbegin))
                        .count();
      ICHECK(period < timeout) << "Failed to connect to server" << addr.AsString();
      LOG(WARNING) << "Cannot connect to tracker " << addr.AsString() << " retry in "
                   << retry_period << " seconds.";
      std::this_thread::sleep_for(std::chrono::seconds(retry_period));
    }
  }

  /*!
   * \brief Generate a random key.
   * \param prefix The string prefix.
   * \return cmap The conflict map set.
   */
  std::string RandomKey(const std::string& prefix, const std::set<std::string>& cmap) {
    if (!cmap.empty()) {
      while (true) {
        std::string key = prefix + std::to_string(dis_(gen_));
        if (cmap.find(key) == cmap.end()) {
          return key;
        }
      }
    }
    return prefix + std::to_string(dis_(gen_));
  }

  std::string tracker_addr_;
  std::string key_;
  std::string custom_addr_;
  support::TCPSocket tracker_sock_;
  std::set<std::string> old_keyset_;
  std::mt19937 gen_;
  std::uniform_real_distribution<float> dis_;
};

/*!
 * \brief RPCServer RPC Server class.
 */
class RPCServer {
 public:
  /*!
   * \brief RPCServer RPC Server class constructor
   * \param host The hostname of the server, Default=0.0.0.0
   * \param port The port of the RPC, Default=9090
   * \param port_end The end search port of the RPC, Default=9099
   * \param tracker_addr The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
   * \param key The key used to identify the device type in tracker. Default=""
   * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
   */
  RPCServer(std::string host, int port, int port_end, std::string tracker_addr, std::string key,
            std::string custom_addr)
      : host_(std::move(host)),
        port_(port),
        my_port_(0),
        port_end_(port_end),
        tracker_addr_(std::move(tracker_addr)),
        key_(std::move(key)),
        custom_addr_(std::move(custom_addr)) {}

  /*!
   * \brief Destructor.
   */
  ~RPCServer() {
    try {
      // Free the resources
      listen_sock_.Close();
    } catch (...) {
    }
  }

  /*!
   * \brief Start Creates the RPC listen process and execution.
   */
  void Start() {
    listen_sock_.Create();
    my_port_ = listen_sock_.TryBindHost(host_, port_, port_end_);
    LOG(INFO) << "bind to " << host_ << ":" << my_port_;
    listen_sock_.Listen(1);
    std::future<void> proc = std::future<void>(std::async(std::launch::async, &RPCServer::ListenLoopProc, this));
    proc.get();
    // Close the listen socket
    listen_sock_.Close();
  }

 private:
  /*!
   * \brief ListenLoopProc The listen process.
   */
  void ListenLoopProc() {
    TrackerClient tracker(tracker_addr_, key_, custom_addr_);
    while (true) {
      support::TCPSocket conn;
      support::SockAddr addr("0.0.0.0", 0);
      std::string opts;
      try {
        // step 1: setup tracker and report to tracker
        tracker.TryConnect();
        // step 2: wait for in-coming connections
        AcceptConnection(&tracker, &conn, &addr, &opts);
      } catch (const char* msg) {
        LOG(WARNING) << "Socket exception: " << msg;
        // close tracker resource
        tracker.Close();
        continue;
      } catch (const std::exception& e) {
        // close tracker resource
        tracker.Close();
        LOG(WARNING) << "Exception standard: " << e.what();
        continue;
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      ServerLoopProc(conn, addr);
      auto dur = std::chrono::high_resolution_clock::now() - start_time;

      LOG(INFO) << "Serve Time " << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << "ms";

      // close from our side.
      LOG(INFO) << "Socket Connection Closed";
      conn.Close();
    }
  }

  /*!
   * \brief AcceptConnection Accepts the RPC Server connection.
   * \param tracker Tracker details.
   * \param conn_sock New connection information.
   * \param addr New connection address information.
   * \param opts Parsed options for socket
   * \param ping_period Timeout for select call waiting
   */
  void AcceptConnection(TrackerClient* tracker, support::TCPSocket* conn_sock,
                        support::SockAddr* addr, std::string* opts, int ping_period = 2) {
    std::set<std::string> old_keyset;
    std::string matchkey;

    // Report resource to tracker and get key
    tracker->ReportResourceAndGetKey(my_port_, &matchkey);

    while (true) {
      tracker->WaitConnectionAndUpdateKey(listen_sock_, my_port_, ping_period, &matchkey);
      support::TCPSocket conn = listen_sock_.Accept(addr);

      int code = kRPCMagic;
      ICHECK_EQ(conn.RecvAll(&code, sizeof(code)), sizeof(code));
      if (code != kRPCMagic) {
        conn.Close();
        LOG(FATAL) << "Client connected is not TVM RPC server";
        continue;
      }

      int keylen = 0;
      ICHECK_EQ(conn.RecvAll(&keylen, sizeof(keylen)), sizeof(keylen));

      const char* CLIENT_HEADER = "client:";
      const char* SERVER_HEADER = "server:";
      std::string expect_header = CLIENT_HEADER + matchkey;
      std::string server_key = SERVER_HEADER + key_;
      if (size_t(keylen) < expect_header.length()) {
        conn.Close();
        LOG(INFO) << "Wrong client header length";
        continue;
      }

      ICHECK_NE(keylen, 0);
      std::string remote_key;
      remote_key.resize(keylen);
      ICHECK_EQ(conn.RecvAll(&remote_key[0], keylen), keylen);

      std::stringstream ssin(remote_key);
      std::string arg0;
      ssin >> arg0;

      if (arg0 != expect_header) {
        code = kRPCMismatch;
        ICHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
        conn.Close();
        LOG(WARNING) << "Mismatch key from" << addr->AsString();
        continue;
      } else {
        code = kRPCSuccess;
        ICHECK_EQ(conn.SendAll(&code, sizeof(code)), sizeof(code));
        keylen = int(server_key.length());
        ICHECK_EQ(conn.SendAll(&keylen, sizeof(keylen)), sizeof(keylen));
        ICHECK_EQ(conn.SendAll(server_key.c_str(), keylen), keylen);
        LOG(INFO) << "Connection success " << addr->AsString();
        ssin >> *opts;
        *conn_sock = conn;
        return;
      }
    }
  }

  /*!
   * \brief ServerLoopProc The Server loop process.
   * \param sock The socket information
   * \param addr The socket address information
   */
  static void ServerLoopProc(support::TCPSocket sock, support::SockAddr addr) {
    // Server loop
    RPCServerLoop(int(sock.sockfd));
    LOG(INFO) << "Finish serving " << addr.AsString();
  }

  std::string host_;
  int port_;
  int my_port_;
  int port_end_;
  std::string tracker_addr_;
  std::string key_;
  std::string custom_addr_;
  support::TCPSocket listen_sock_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_APPS_IOS_RPC_SERVER_H_
