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

#ifndef TVM_APPS_IOS_RPC_ENV_H_
#define TVM_APPS_IOS_RPC_ENV_H_

namespace tvm {
namespace runtime {

/*!
 * Environment handler
 * Serves temp directory with resources
 */
struct RPCEnv {
 public:
  RPCEnv() {
    NSString* path = NSTemporaryDirectory();
    base_ = [path UTF8String];
    if (base_[base_.length() - 1] != '/') {
      base_ = base_ + '/';
    }
  }
  
  // Get path to file by name
  std::string GetPath(const std::string& file_name) { return base_ + file_name; }
  
  // Clean up envinroment
  void CleanUp() const {
    CleanDir(base_);
  }
  
 private:
  std::string base_;
  
  /*!
   * \brief Browse folder and retern list of content
   * \param dirname path to fodler to browse through
   */
  static std::vector<std::string> ListDir(const std::string& dirname) {
    std::vector<std::string> vec;
    DIR* dp = opendir(dirname.c_str());
    if (dp == nullptr) {
      int errsv = errno;
      LOG(FATAL) << "ListDir " << dirname << " error: " << strerror(errsv);
    }
    dirent* d;
    while ((d = readdir(dp)) != nullptr) {
      std::string filename = d->d_name;
      if (filename != "." && filename != "..") {
        std::string f = dirname;
        if (f[f.length() - 1] != '/') {
          f += '/';
        }
        f += d->d_name;
        vec.push_back(f);
      }
    }
    closedir(dp);
    return vec;
  }

  /*!
   * \brief CleanDir Removes the files from the directory
   * \param dirname The name of the directory
   */
  static void CleanDir(const std::string& dirname) {
    auto files = ListDir(dirname);
    for (const auto& filename : files) {
      std::string file_path = dirname + "/";
      file_path += filename;
      const int ret = std::remove(filename.c_str());
      if (ret != 0) {
        LOG(WARNING) << "Remove file " << filename << " failed";
      }
    }
  }
};

}  // namespace runtime
}  // namespace tvm

#endif // TVM_APPS_IOS_RPC_ENV_H_
