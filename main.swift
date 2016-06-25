#if os(OSX) || os(iOS)
import Foundation
import ImageIO
import CoreServices

let width = 512
let height = 512
let featureSize = 24.0

let noise = OpenSimplexNoise()

var pixels = [UInt8](count: width * height * 4, repeatedValue: 0)

for y in 0..<height {
    for x in 0..<width {
        let value = noise.eval(x: Double(x) / featureSize, y: Double(y) / featureSize)
        let rgb = UInt8((value + 1) * 127.5)
        pixels[(y * width + x) * 4 + 0] = 255
        pixels[(y * width + x) * 4  + 1] = rgb
        pixels[(y * width + x) * 4  + 2] = rgb
        pixels[(y * width + x) * 4  + 3] = rgb
    }
}

let render = CGColorRenderingIntent.RenderingIntentDefault
let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.PremultipliedFirst.rawValue)
let providerRef = CGDataProviderCreateWithCFData(NSData(bytes: pixels, length: width * height * 4))
let cgImage = CGImageCreate(width, height, 8, 32, width * 4, rgbColorSpace, bitmapInfo, providerRef, nil, true, render)

if let cgImage = cgImage {
    let url: CFURLRef = NSURL.fileURLWithPath("swiftNoise.png")
    if let destination = CGImageDestinationCreateWithURL(url, kUTTypePNG, 1, nil) {
        CGImageDestinationAddImage(destination, cgImage, nil)
        CGImageDestinationFinalize(destination)
    }
}
#elseif os(Linux)
print("Demo requires OSX / iOS libraries")
#endif
