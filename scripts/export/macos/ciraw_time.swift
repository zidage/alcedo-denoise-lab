// Time of the RAW processing of macOS (Core Image CIRAWFilter) for a comparison with the model.
//
// For each RAW file and each noise reduction setting: a new CIRAWFilter, its output image at
// the full size, rendered by one Metal CIContext into a float RGBA bitmap in host memory
// (linear extended sRGB). 1 run that is not measured, then `repeats` runs. The record has the
// decoder version, the native size, the default noise reduction amounts, and the median time.
//
// Usage: ciraw_time <out.json> <repeats> <raw file>...
// Step: export, cost, and stored result (step 7 of docs/plan.md of alcedo-denoise-lab).

import CoreImage
import Foundation
import Metal

func seconds(_ start: DispatchTime) -> Double {
  Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
}

let arguments = CommandLine.arguments
guard arguments.count >= 4, let repeats = Int(arguments[2]) else {
  print("usage: ciraw_time <out.json> <repeats> <raw file>...")
  exit(2)
}
let device = MTLCreateSystemDefaultDevice()!
let context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false,
                                                     .workingFormat: CIFormat.RGBAh])
let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
var records: [[String: Any]] = []

for path in arguments[3...] {
  let url = URL(fileURLWithPath: path)
  guard let probe = CIRAWFilter(imageURL: url) else {
    records.append(["file": url.lastPathComponent, "error": "CIRAWFilter has no decoder for this file"])
    print("\(url.lastPathComponent): no decoder")
    continue
  }
  let size = probe.nativeSize
  var record: [String: Any] = [
    "file": url.lastPathComponent,
    "native_size": [Int(size.width), Int(size.height)],
    "megapixels": Double(size.width * size.height) / 1e6,
    "decoder_version": probe.decoderVersion.rawValue,
    "supported_decoder_versions": probe.supportedDecoderVersions.map { $0.rawValue },
    "default_luminance_noise_reduction": probe.luminanceNoiseReductionAmount,
    "default_color_noise_reduction": probe.colorNoiseReductionAmount,
    "luminance_noise_reduction_supported": probe.isLuminanceNoiseReductionSupported,
    "camera": (probe.properties[kCGImagePropertyTIFFDictionary as String] as? [String: Any])?[
      kCGImagePropertyTIFFModel as String] ?? "",
  ]
  // The output extent can be larger than the native size (Sony ILCE-7RM3), so the bitmap has
  // the size of the extent.
  let extent = probe.outputImage!.extent
  record["output_extent"] = [Int(extent.width), Int(extent.height)]
  let rowBytes = Int(extent.width) * 16
  var bitmap = [UInt8](repeating: 0, count: rowBytes * Int(extent.height))
  var variants: [[String: Any]] = []
  // default: the amounts of the decoder for this file; off: 0; max: 1.
  for variant in ["default", "off", "max"] {
    var times: [Double] = []
    for run in 0...repeats {
      let start = DispatchTime.now()
      let filter = CIRAWFilter(imageURL: url)!
      if variant == "off" {
        filter.luminanceNoiseReductionAmount = 0
        filter.colorNoiseReductionAmount = 0
      } else if variant == "max" {
        filter.luminanceNoiseReductionAmount = 1
        filter.colorNoiseReductionAmount = 1
      }
      let image = filter.outputImage!
      precondition(image.extent.size == extent.size, "extent changed with the noise reduction")
      bitmap.withUnsafeMutableBytes { buffer in
        context.render(image, toBitmap: buffer.baseAddress!, rowBytes: rowBytes,
                       bounds: image.extent, format: .RGBAf, colorSpace: linear)
      }
      if run > 0 { times.append(seconds(start)) }
    }
    times.sort()
    let median = times[times.count / 2]
    variants.append(["noise_reduction": variant, "seconds": times, "median_seconds": median])
    print("\(url.lastPathComponent) \(Int(size.width))x\(Int(size.height)) \(variant): \(median) s")
  }
  record["runs"] = variants
  records.append(record)
}
let json = try JSONSerialization.data(withJSONObject: ["device": device.name, "records": records],
                                      options: [.prettyPrinted, .sortedKeys])
try json.write(to: URL(fileURLWithPath: arguments[1]))
