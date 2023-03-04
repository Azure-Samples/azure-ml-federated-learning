cwd=$(pwd)
apt-get update
apt-get install -y zip pkg-config build-essential cmake
git clone https://github.com/microsoft/vcpkg.git /vcpkg
chmod a+x /vcpkg/bootstrap-vcpkg.sh
/bin/bash /vcpkg/bootstrap-vcpkg.sh
/vcpkg/vcpkg install --triplet=x64-linux --x-buildtrees-root=/vcpkg/buildtrees --x-install-root=/vcpkg/installed --x-packages-root=/vcpkg/packages
/vcpkg/vcpkg list
git clone https://github.com/microsoft/APSI.git /APSI && cd /APSI && git checkout 2dff8dcd39c361527ea3b320f87cb8e71dd4f777 && mkdir build
sed -i 's/SEAL 4 QUIET REQUIRED/SEAL REQUIRED/g' CMakeLists.txt && cd build
cmake .. -DCMAKE_FIND_DEBUG_MODE=ON -DAPSI_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake
make -j$(nproc) && make install
cd $cwd
ls /APSI/build/bin -la