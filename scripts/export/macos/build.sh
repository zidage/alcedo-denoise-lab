#!/bin/sh
# Build ort_macos_measure against each ONNX Runtime package in ../pkg.
set -eu
cd "$(dirname "$0")"
for version in 1.24.4 1.32.0; do
  root="../pkg/ort-$version"
  # The tool links the dylib through its rpath. A changed dylib needs a new ad hoc signature, or
  # macOS kills the process at load (status 137).
  install_name_tool -id @rpath/libonnxruntime.dylib "$root/lib/libonnxruntime.dylib" 2>/dev/null || true
  codesign -f -s - "$root/lib/libonnxruntime.dylib"
  xcrun clang++ -std=c++17 -O2 -fobjc-arc -x objective-c++ ort_macos_measure.mm \
    -I "$root/include" -L "$root/lib" -lonnxruntime -framework Foundation \
    -Wl,-rpath,"$(cd "$root/lib" && pwd)" -o "ort_macos_measure-$version"
  echo "built ort_macos_measure-$version"
done
