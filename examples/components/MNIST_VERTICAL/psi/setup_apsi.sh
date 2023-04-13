cwd=$(pwd)
apt-get update
apt-get install -y zip pkg-config build-essential cmake
git clone https://github.com/microsoft/vcpkg.git /vcpkg
chmod a+x /vcpkg/bootstrap-vcpkg.sh
/bin/bash /vcpkg/bootstrap-vcpkg.sh
/vcpkg/vcpkg install --triplet=x64-linux --x-buildtrees-root=/vcpkg/buildtrees --x-install-root=/vcpkg/installed --x-packages-root=/vcpkg/packages
/vcpkg/vcpkg list
git clone https://github.com/microsoft/APSI.git /APSI && cd /APSI && git checkout 0bd91cbca8c0ae39a28948c337a96c983933cb79 && mkdir build
sed -i 's/SEAL 4 QUIET REQUIRED/SEAL REQUIRED/g' CMakeLists.txt && cd build
cmake .. -DCMAKE_FIND_DEBUG_MODE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake
make -j$(nproc) && make install
ls /APSI/build/bin -la

cd $cwd/SymmetricPSI && mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake -DAPSI_ROOT=/APSI/build -DVCPKG_TARGET_TRIPLET=x64-linux -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp ./SymmetricPSI.so $cwd

cd $cwd