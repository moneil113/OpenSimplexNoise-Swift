/// OpenSimplex Noise in Swift
/// Ported to Swift by Matthew O'Neil based on Kurt Spencer's orignial Java implementation

#if os(OSX) || os(iOS)
import Darwin
#elseif os(Linux)
import Glibc
#endif

let DEFAULT_SEED: Int64 = 0

public class OpenSimplexNoise {
    private let STRETCH_CONSTANT_2D = -0.211324865405187    // (1 / sqrt(2 + 1) - 1) / 2
    private let SQUISH_CONSTANT_2D = 0.366025403784439      // (sqrt( 2 + 1) - 1) / 2
    private let STRETCH_CONSTANT_3D = -1.0 / 6              // (1 / sqrt( 3 + 1) - 1) / 3
    private let SQUISH_CONSTANT_3D = 1.0 / 3                // (sqrt(3 + 1) - 1) / 3
    private let STRETCH_CONSTANT_4D = -0.138196601125011    // (1 / sqrt(4 + 1) - 1) / 4
    private let SQUISH_CONSTANT_4D = 0.309016994374947      // (sqrt(4 + 1) - 1) / 4

    private let NORM_CONSTANT_2D = 47
    private let NORM_CONSTANT_3D = 103
    private let NORM_CONSTANT_4D = 30

    private var perm: [Int16]
    private var permGradIndex3D: [Int16]

    public convenience init() {
        self.init(seed: DEFAULT_SEED)
    }

    public init(perm: [Int16]) {
        self.perm = perm
        permGradIndex3D = [Int16](count: 256, repeatedValue: 0)

        for i in 0..<256 {
            //Since 3D has 24 gradients, simple bitmask won't work, so precompute modulo array.
            permGradIndex3D[i] = (perm[i] % Int16(gradients3D.count) / 3) * 3
        }
    }

    /// Initialises the class using a permutation array generated from 64-bit seed.
    /// Generates a proper permutation (i.e. doesn't merely perform N successive pair swaps on a base array)
    /// Uses a simple 64-bit LCG.
    public init(seed _seed: Int64) {
        perm = [Int16](count: 256, repeatedValue: 0)
        permGradIndex3D = [Int16](count: 256, repeatedValue: 0)
        var source: [Int16] = [Int16](count: 256, repeatedValue: 0)
        for i in 0..<256 {
            source[i] = Int16(i)
        }
        var seed = _seed &* Int64(6364136223846793005) &+ Int64(1442695040888963407)
        seed = seed &* Int64(6364136223846793005) &+ Int64(1442695040888963407)
        seed = seed &* Int64(6364136223846793005) &+ Int64(1442695040888963407)
        for i in (0..<256).reverse() {
            seed = seed &* Int64(6364136223846793005) &+ Int64(1442695040888963407)
            var r = Int((seed + 31) % (i + 1))
            if (r < 0) {
                r += i + 1
            }
            perm[i] = source[r]
            permGradIndex3D[i] = (perm[i] % Int16(gradients3D.count / 3)) * 3
            source[r] = source[i]
        }
    }

    /// 2D OpenSimplex Noise.
    public func eval(x x: Double, y: Double) -> Double {
        //Place input coordinates onto grid.
        let stretchOffset = (x + y) * STRETCH_CONSTANT_2D
        let xs = x + stretchOffset
        let ys = y + stretchOffset

        //Floor to get grid coordinates of rhomubs (stretched square) super-cell origin.
        var xsb = fastFloor(xs)
        var ysb = fastFloor(ys)

        //Skew out to get actual coordinates of rhuombus origin. We'll need these later.
        let squishOffset = Double(xsb + ysb) * SQUISH_CONSTANT_2D
        let xb = Double(xsb) + squishOffset
        let yb = Double(ysb) + squishOffset

        //Compute grid coordinates relative to rhombus origin.
        let xins = xs - Double(xsb)
        let yins = ys - Double(ysb)

        //Sum those together to get a value that determines which region we're in.
        let inSum = xins + yins

        //Positions relative to origin point.
        var dx0 = x - xb
        var dy0 = y - yb

        //We'll be defining these inside the next block and using them afterwards.
        var dx_ext: Double, dy_ext: Double
        var xsv_ext: Int, ysv_ext: Int

        var value: Double = 0

        //Contribution (1,0)
        let dx1 = dx0 - 1 - SQUISH_CONSTANT_2D
        let dy1 = dy0 - 0 - SQUISH_CONSTANT_2D
        var attn1 = 2 - dx1 * dx1 - dy1 * dy1
        if (attn1 > 0) {
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, dx: dx1, dy: dy1)
        }

        //Contribution (0,1)
        let dx2 = dx0 - 0 - SQUISH_CONSTANT_2D
        let dy2 = dy0 - 1 - SQUISH_CONSTANT_2D
        var attn2 = 2 - dx2 * dx2 - dy2 * dy2
        if (attn2 > 0) {
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, dx: dx2, dy: dy2)
        }

        if (inSum <= 1) { //We're inside the triangle (2-Simplex) at 0,0)
            let zins = 1 - inSum
            if (zins > xins || zins > yins) { //(0,0) is one fothe the closest two triangular vertices
                if (xins > yins) {
                    xsv_ext = xsb + 1
                    ysv_ext = ysb - 1
                    dx_ext = dx0 - 1
                    dy_ext = dy0 + 1
                }
                else {
                    xsv_ext = xsb - 1
                    ysv_ext = ysb + 1
                    dx_ext = dx0 + 1
                    dy_ext = dy0 - 1
                }
            }
            else { //(1,0) and (0,1) are the closest two vertices
                xsv_ext = xsb + 1
                ysv_ext = ysb + 1
                dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
                dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT_2D
            }
        }
        else { //We're inside the triangle (2-Simplex) at (1,1)
            let zins = 2 - inSum
            if (zins < xins || zins < yins) { //(0,0) is one of the closest two triangular vertices
                if (xins > yins) {
                    xsv_ext = xsb + 2
                    ysv_ext = ysb + 0
                    dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT_2D
                    dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT_2D
                }
                else { // (1,0) and (0,1) are the closest two vertices.
                    dx_ext = dx0
                    dy_ext = dy0
                    xsv_ext = xsb
                    ysv_ext = ysb
                }
            }
            else {
                dx_ext = dx0
                dy_ext = dy0
                xsv_ext = xsb
                ysv_ext = ysb
            }
            xsb += 1
            ysb += 1
            dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
            dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT_2D
        }

        //Contribution (0,0) or (1,1)
        var attn0 = 2 - dx0 * dx0 - dy0 * dy0
        if (attn0 > 0) {
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate(xsb: xsb, ysb: ysb, dx: dx0, dy: dy0)
        }

        //Extra Vertex
        var attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
        if (attn_ext > 0) {
            attn_ext *= attn_ext
            value += attn_ext * attn_ext * extrapolate(xsb: xsv_ext, ysb: ysv_ext, dx: dx_ext, dy: dy_ext)
        }

        return value / Double(NORM_CONSTANT_2D)
    }

    /// 3D OpenSimplex Noise.
    public func eval(x x: Double, y: Double, z: Double) -> Double {
        //Place input coordinates on simplectic honeycomb.
        let stretchOffset = (x + y + z) * STRETCH_CONSTANT_3D
        let xs = x + stretchOffset
        let ys = y + stretchOffset
        let zs = z + stretchOffset

        //Floor to get simplectic honeycomb coordinates of rhombohedron (stretched cube) super-cell origin.
        let xsb = fastFloor(xs)
        let ysb = fastFloor(ys)
        let zsb = fastFloor(zs)

        //Skew out to get actual coordinates of rhombohedron origin. We'll need these later.
        let squishOffset = Double(xsb + ysb + zsb) * SQUISH_CONSTANT_3D
        let xb = Double(xsb) + squishOffset
        let yb = Double(ysb) + squishOffset
        let zb = Double(zsb) + squishOffset

        //Compute simplectic honeycomb coordinates relative to rhombohedral origin.
        let xins = xs - Double(xsb)
        let yins = ys - Double(ysb)
        let zins = zs - Double(zsb)

        //Sum those together to get a value that determines which region we're in.
        let inSum = xins + yins + zins

        //Positions relative to origin point.
        var dx0 = x - xb
        var dy0 = y - yb
        var dz0 = z - zb

        //We'll be defining these inside the next block and using them afterwards.
        var dx_ext0: Double, dy_ext0: Double, dz_ext0: Double
        var dx_ext1: Double, dy_ext1: Double, dz_ext1: Double
        var xsv_ext0: Int, ysv_ext0: Int, zsv_ext0: Int
        var xsv_ext1: Int, ysv_ext1: Int, zsv_ext1: Int

        var value: Double = 0
        if (inSum <= 1) { //We're inside the tetrahedron (3-Simplex) at (0,0,0)

            //Determine which two of (0,0,1), (0,1,0), (1,0,0) are closest.
            var aPoint = 0x01
            var aScore = xins
            var bPoint = 0x02
            var bScore = yins
            if (aScore >= bScore && zins > bScore) {
                bScore = zins
                bPoint = 0x04
            }
            else if (aScore < bScore && zins > aScore) {
                aScore = zins
                aPoint = 0x04
            }

            //Now we determine the two lattice points not part of the tetrahedron that may contribute.
            //This depends on the closest two tetrahedral vertices, including (0,0,0)
            let wins = 1 - inSum
            if (wins > aScore || wins > bScore) { //(0,0,0) is one of the closest two tetrahedral vertices.
                let c = (bScore > aScore ? bPoint : aPoint) //Our other closest vertex is the closest out of a and b.

                if ((c & 0x01) == 0) {
                    xsv_ext0 = xsb - 1
                    xsv_ext1 = xsb
                    dx_ext0 = dx0 + 1
                    dx_ext1 = dx0
                }
                else {
                    xsv_ext1 = xsb + 1
                    xsv_ext0 = xsv_ext1
                    dx_ext1 = dx0 - 1
                    dx_ext0 = dx_ext1
                }

                if ((c & 0x02) == 0) {
                    ysv_ext1 = ysb
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0
                    dy_ext0 = dy_ext1
                    if ((c & 0x01) == 0) {
                        ysv_ext1 -= 1
                        dy_ext1 += 1
                    }
                    else {
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                    }
                }
                else {
                    ysv_ext1 = ysb + 1
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0 - 1
                    dy_ext0 = dy_ext1
                }

                if ((c & 0x04) == 0) {
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0
                    dz_ext1 = dz0 + 1
                }
                else {
                    zsv_ext1 = zsb + 1
                    zsv_ext0 = zsv_ext1
                    dz_ext1 = dz0 - 1
                    dz_ext0 = dz_ext1
                }
            }
            else { //(0,0,0) is not one of the closest two tetrahedral vertices.
                let c = UInt8(aPoint | bPoint) //Our two extra vertices are determined by the closest two.

                if ((c & 0x01) == 0) {
                    xsv_ext0 = xsb
                    xsv_ext1 = xsb - 1
                    dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_3D
                }
                else {
                    xsv_ext1 = xsb + 1
                    xsv_ext0 = xsv_ext1
                    dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D
                }

                if ((c & 0x02) == 0) {
                    ysv_ext0 = ysb
                    ysv_ext1 = ysb - 1
                    dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 + 1 - SQUISH_CONSTANT_3D
                }
                else {
                    ysv_ext1 = ysb + 1
                    ysv_ext0 = ysv_ext1
                    dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D
                }

                if ((c & 0x04) == 0) {
                    zsv_ext0 = zsb
                    zsv_ext1 = zsb - 1
                    dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 + 1 - SQUISH_CONSTANT_3D
                }
                else {
                    zsv_ext1 = zsb + 1
                    zsv_ext0 = zsv_ext1
                    dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D
                }
            }

            //Contribution (0,0,0)
            var attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
            if (attn0 > 0) {
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 0, dx: dx0, dy: dy0, dz: dz0)
            }

            //Contribution (1,0,0)
            let dx1 = dx0 - 1 - SQUISH_CONSTANT_3D
            let dy1 = dy0 - 0 - SQUISH_CONSTANT_3D
            let dz1 = dz0 - 0 - SQUISH_CONSTANT_3D
            var attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if (attn1 > 0) {
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 0, dx: dx1, dy: dy1, dz: dz1)
            }

            //Contribution (0,1,0)
            let dx2 = dx0 - 0 - SQUISH_CONSTANT_3D
            let dy2 = dy0 - 1 - SQUISH_CONSTANT_3D
            let dz2 = dz1
            var attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if (attn2 > 0) {
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 0, dx: dx2, dy: dy2, dz: dz2)
            }

            //Contribution (0,0,1)
            let dx3 = dx2
            let dy3 = dy1
            let dz3 = dz0 - 1 - SQUISH_CONSTANT_3D
            var attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if (attn3 > 0) {
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 1, dx: dx3, dy: dy3, dz: dz3)
            }
        }
        else if (inSum >= 2) { //We're inside the tetrahedron (3-Simplex) at (1,1,1)

            //Determine which two tetrahedral vertices are the closest, out of (1,1,0), (1,0,1), (0,1,1) but not (1,1,1).
            var aPoint = 0x06
            var aScore = xins
            var bPoint = 0x05
            var bScore = yins
            if (aScore <= bScore && zins < bScore) {
                bScore = zins
                bPoint = 0x03
            }
            else if (aScore > bScore && zins < aScore) {
                aScore = zins
                aPoint = 0x03
            }

            //Now we determine the two lattice points not part of the tetrahedron that may contribute.
            //This depends on the closest two tetrahedral vertices, including (1,1,1)
            let wins = 3 - inSum
            if (wins < aScore || wins < bScore) { //(1,1,1) is one of the closest two tetrahedral vertices.
                let c = (bScore < aScore ? bPoint : aPoint) //Our other closest vertex is the closest out of a and b.

                if ((c & 0x01) != 0) {
                    xsv_ext0 = xsb + 2
                    xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D
                }
                else {
                    xsv_ext1 = xsb
                    xsv_ext0 = xsv_ext1
                    dx_ext1 = dx0 - 3 * SQUISH_CONSTANT_3D
                    dx_ext0 = dx_ext1
                }

                if ((c & 0x02) != 0) {
                    ysv_ext1 = ysb + 1
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D
                    dy_ext0 = dy_ext1
                    if ((c & 0x01) != 0) {
                        ysv_ext1 += 1
                        dy_ext1 -= 1
                    }
                    else {
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                    }
                }
                else {
                    ysv_ext1 = ysb
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0 - 3 * SQUISH_CONSTANT_3D
                    dy_ext0 = dy_ext1
                }

                if ((c & 0x04) != 0) {
                    zsv_ext0 = zsb + 1
                    zsv_ext1 = zsb + 2
                    dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 - 3 * SQUISH_CONSTANT_3D
                }
                else {
                    zsv_ext1 = zsb
                    zsv_ext0 = zsv_ext1
                    dz_ext1 = dz0 - 3 * SQUISH_CONSTANT_3D
                    dz_ext0 = dz_ext1
                }
            }
            else { //(1,1,1) is not one of the closest two tetrahedral vertices.
                let c = UInt8((aPoint & bPoint)) //Our two extra vertices are determined by the closest two.

                if ((c & 0x01) != 0) {
                    xsv_ext0 = xsb + 1
                    xsv_ext1 = xsb + 2
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D
                }
                else {
                    xsv_ext1 = xsb
                    xsv_ext0 = xsv_ext1
                    dx_ext0 = dx0 - SQUISH_CONSTANT_3D
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                }

                if ((c & 0x02) != 0) {
                    ysv_ext0 = ysb + 1
                    ysv_ext1 = ysb + 2
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D
                }
                else {
                    ysv_ext1 = ysb
                    ysv_ext0 = ysv_ext1
                    dy_ext0 = dy0 - SQUISH_CONSTANT_3D
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                }

                if ((c & 0x04) != 0) {
                    zsv_ext0 = zsb + 1
                    zsv_ext1 = zsb + 2
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D
                }
                else {
                    zsv_ext1 = zsb
                    zsv_ext0 = zsv_ext1
                    dz_ext0 = dz0 - SQUISH_CONSTANT_3D
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
                }
            }

            //Contribution (1,1,0)
            let dx3 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
            let dy3 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
            let dz3 = dz0 - 0 - 2 * SQUISH_CONSTANT_3D
            var attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if (attn3 > 0) {
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 0, dx: dx3, dy: dy3, dz: dz3)
            }

            //Contribution (1,0,1)
            let dx2 = dx3
            let dy2 = dy0 - 0 - 2 * SQUISH_CONSTANT_3D
            let dz2 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
            var attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if (attn2 > 0) {
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 1, dx: dx2, dy: dy2, dz: dz2)
            }

            //Contribution (0,1,1)
            let dx1 = dx0 - 0 - 2 * SQUISH_CONSTANT_3D
            let dy1 = dy3
            let dz1 = dz2
            var attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if (attn1 > 0) {
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 1, dx: dx1, dy: dy1, dz: dz1)
            }

            //Contribution (1,1,1)
            dx0 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D
            dy0 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D
            dz0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D
            var attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
            if (attn0 > 0) {
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 1, dx: dx0, dy: dy0, dz: dz0)
            }
        }
        else { //We're inside the octahedron (Rectified 3-Simplex) in between.
            var aScore: Double
            var aPoint: UInt8
            var aIsFurtherSide: Bool
            var bScore: Double
            var bPoint: UInt8
            var bIsFurtherSide: Bool

            //Decide between point (0,0,1) and (1,1,0) as closest
            let p1 = xins + yins
            if (p1 > 1) {
                aScore = p1 - 1
                aPoint = 0x03
                aIsFurtherSide = true
            }
            else {
                aScore = 1 - p1
                aPoint = 0x04
                aIsFurtherSide = false
            }

            //Decide between point (0,1,0) and (1,0,1) as closest
            let p2 = xins + zins
            if (p2 > 1) {
                bScore = p2 - 1
                bPoint = 0x05
                bIsFurtherSide = true
            }
            else {
                bScore = 1 - p2
                bPoint = 0x02
                bIsFurtherSide = false
            }

            //The closest out of the two (1,0,0) and (0,1,1) will replace the furthest out of the two decided above, if closer.
            let p3 = yins + zins
            if (p3 > 1) {
                let score = p3 - 1
                if (aScore <= bScore && aScore < score) {
                    aScore = score
                    aPoint = 0x06
                    aIsFurtherSide = true
                }
                else if (aScore > bScore && bScore < score) {
                    bScore = score
                    bPoint = 0x06
                    bIsFurtherSide = true
                }
            }
            else {
                let score = 1 - p3
                if (aScore <= bScore && aScore < score) {
                    aScore = score
                    aPoint = 0x01
                    aIsFurtherSide = false
                }
                else if (aScore > bScore && bScore < score) {
                    bScore = score
                    bPoint = 0x01
                    bIsFurtherSide = false
                }
            }

            //Where each of the two closest points are determines how the extra two vertices are calculated.
            if (aIsFurtherSide == bIsFurtherSide) {
                if (aIsFurtherSide) { //Both closest points on (1,1,1) side

                    //One of the two extra points is (1,1,1)
                    dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb + 1
                    ysv_ext0 = ysb + 1
                    zsv_ext0 = zsb + 1

                    //Other extra point is based on the shared axis.
                    let c = UInt8(aPoint & bPoint)
                    if ((c & 0x01) != 0) {
                        dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb + 2
                        ysv_ext1 = ysb
                        zsv_ext1 = zsb
                    }
                    else if ((c & 0x02) != 0) {
                        dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb
                        ysv_ext1 = ysb + 2
                        zsv_ext1 = zsb
                    }
                    else {
                        dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb
                        ysv_ext1 = ysb
                        zsv_ext1 = zsb + 2
                    }
                }
                else {//Both closest points on (0,0,0) side

                    //One of the two extra points is (0,0,0)
                    dx_ext0 = dx0
                    dy_ext0 = dy0
                    dz_ext0 = dz0
                    xsv_ext0 = xsb
                    ysv_ext0 = ysb
                    zsv_ext0 = zsb

                    //Other extra point is based on the omitted axis.
                    let c = UInt8(aPoint | bPoint)
                    if ((c & 0x01) == 0) {
                        dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb - 1
                        ysv_ext1 = ysb + 1
                        zsv_ext1 = zsb + 1
                    }
                    else if ((c & 0x02) == 0) {
                        dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 + 1 - SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb + 1
                        ysv_ext1 = ysb - 1
                        zsv_ext1 = zsb + 1
                    }
                    else {
                        dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_3D
                        dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_3D
                        dz_ext1 = dz0 + 1 - SQUISH_CONSTANT_3D
                        xsv_ext1 = xsb + 1
                        ysv_ext1 = ysb + 1
                        zsv_ext1 = zsb - 1
                    }
                }
            }
            else { //One point on (0,0,0) side, one point on (1,1,1) side
                var c1: UInt8, c2: UInt8
                if (aIsFurtherSide) {
                    c1 = aPoint
                    c2 = bPoint
                }
                else {
                    c1 = bPoint
                    c2 = aPoint
                }

                //One contribution is a permutation of (1,1,-1)
                if ((c1 & 0x01) == 0) {
                    dx_ext0 = dx0 + 1 - SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb - 1
                    ysv_ext0 = ysb + 1
                    zsv_ext0 = zsb + 1
                }
                else if ((c1 & 0x02) == 0) {
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 + 1 - SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 - 1 - SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb + 1
                    ysv_ext0 = ysb - 1
                    zsv_ext0 = zsb + 1
                }
                else {
                    dx_ext0 = dx0 - 1 - SQUISH_CONSTANT_3D
                    dy_ext0 = dy0 - 1 - SQUISH_CONSTANT_3D
                    dz_ext0 = dz0 + 1 - SQUISH_CONSTANT_3D
                    xsv_ext0 = xsb + 1
                    ysv_ext0 = ysb + 1
                    zsv_ext0 = zsb - 1
                }

                //One contribution is a permutation of (0,0,2)
                dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_3D
                dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_3D
                dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_3D
                xsv_ext1 = xsb
                ysv_ext1 = ysb
                zsv_ext1 = zsb
                if ((c2 & 0x01) != 0) {
                    dx_ext1 -= 2
                    xsv_ext1 += 2
                }
                else if ((c2 & 0x02) != 0) {
                    dy_ext1 -= 2
                    ysv_ext1 += 2
                }
                else {
                    dz_ext1 -= 2
                    zsv_ext1 += 2
                }
            }

            //Contribution (1,0,0)
            let dx1 = dx0 - 1 - SQUISH_CONSTANT_3D
            let dy1 = dy0 - 0 - SQUISH_CONSTANT_3D
            let dz1 = dz0 - 0 - SQUISH_CONSTANT_3D
            var attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
            if (attn1 > 0) {
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 0, dx: dx1, dy: dy1, dz: dz1)
            }

            //Contribution (0,1,0)
            let dx2 = dx0 - 0 - SQUISH_CONSTANT_3D
            let dy2 = dy0 - 1 - SQUISH_CONSTANT_3D
            let dz2 = dz1
            var attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
            if (attn2 > 0) {
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 0, dx: dx2, dy: dy2, dz: dz2)
            }

            //Contribution (0,0,1)
            let dx3 = dx2
            let dy3 = dy1
            let dz3 = dz0 - 1 - SQUISH_CONSTANT_3D
            var attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
            if (attn3 > 0) {
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 1, dx: dx3, dy: dy3, dz: dz3)
            }

            //Contribution (1,1,0)
            let dx4 = dx0 - 1 - 2 * SQUISH_CONSTANT_3D
            let dy4 = dy0 - 1 - 2 * SQUISH_CONSTANT_3D
            let dz4 = dz0 - 0 - 2 * SQUISH_CONSTANT_3D
            var attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4
            if (attn4 > 0) {
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 0, dx: dx4, dy: dy4, dz: dz4)
            }

            //Contribution (1,0,1)
            let dx5 = dx4
            let dy5 = dy0 - 0 - 2 * SQUISH_CONSTANT_3D
            let dz5 = dz0 - 1 - 2 * SQUISH_CONSTANT_3D
            var attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5
            if (attn5 > 0) {
                attn5 *= attn5
                value += attn5 * attn5 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 1, dx: dx5, dy: dy5, dz: dz5)
            }

            //Contribution (0,1,1)
            let dx6 = dx0 - 0 - 2 * SQUISH_CONSTANT_3D
            let dy6 = dy4
            let dz6 = dz5
            var attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6
            if (attn6 > 0) {
                attn6 *= attn6
                value += attn6 * attn6 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 1, dx: dx6, dy: dy6, dz: dz6)
            }
        }

        //First extra vertex
        var attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0
        if (attn_ext0 > 0) {
            attn_ext0 *= attn_ext0
            value += attn_ext0 * attn_ext0 * extrapolate(xsb: xsv_ext0, ysb: ysv_ext0, zsb: zsv_ext0, dx: dx_ext0, dy: dy_ext0, dz: dz_ext0)
        }

        //Second extra vertex
        var attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1
        if (attn_ext1 > 0) {
            attn_ext1 *= attn_ext1
            value += attn_ext1 * attn_ext1 * extrapolate(xsb: xsv_ext1, ysb: ysv_ext1, zsb: zsv_ext1, dx: dx_ext1, dy: dy_ext1, dz: dz_ext1)
        }

        return value / Double(NORM_CONSTANT_3D)
    }

    /// 4D OpenSimplex Noise.
    public func eval(x x: Double, y: Double, z: Double, w: Double) -> Double {

        //Place input coordinates on simplectic honeycomb.
        let stretchOffset = (x + y + z + w) * STRETCH_CONSTANT_4D
        let xs = x + stretchOffset
        let ys = y + stretchOffset
        let zs = z + stretchOffset
        let ws = w + stretchOffset

        //Floor to get simplectic honeycomb coordinates of rhombo-hypercube super-cell origin.
        let xsb = fastFloor(xs)
        let ysb = fastFloor(ys)
        let zsb = fastFloor(zs)
        let wsb = fastFloor(ws)

        //Skew out to get actual coordinates of stretched rhombo-hypercube origin. We'll need these later.
        let squishOffset = Double(xsb + ysb + zsb + wsb) * SQUISH_CONSTANT_4D
        let xb = Double(xsb) + squishOffset
        let yb = Double(ysb) + squishOffset
        let zb = Double(zsb) + squishOffset
        let wb = Double(wsb) + squishOffset

        //Compute simplectic honeycomb coordinates relative to rhombo-hypercube origin.
        let xins = xs - Double(xsb)
        let yins = ys - Double(ysb)
        let zins = zs - Double(zsb)
        let wins = ws - Double(wsb)

        //Sum those together to get a value that determines which region we're in.
        let inSum = xins + yins + zins + wins

        //Positions relative to origin point.
        var dx0 = x - xb
        var dy0 = y - yb
        var dz0 = z - zb
        var dw0 = w - wb

        //We'll be defining these inside the next block and using them afterwards.
        var dx_ext0: Double, dy_ext0: Double, dz_ext0: Double, dw_ext0: Double
        var dx_ext1: Double, dy_ext1: Double, dz_ext1: Double, dw_ext1: Double
        var dx_ext2: Double, dy_ext2: Double, dz_ext2: Double, dw_ext2: Double
        var xsv_ext0: Int, ysv_ext0: Int, zsv_ext0: Int, wsv_ext0: Int
        var xsv_ext1: Int, ysv_ext1: Int, zsv_ext1: Int, wsv_ext1: Int
        var xsv_ext2: Int, ysv_ext2: Int, zsv_ext2: Int, wsv_ext2: Int

        var value: Double = 0
        if (inSum <= 1) { //We're inside the pentachoron (4-Simplex) at (0,0,0,0)

            //Determine which two of (0,0,0,1), (0,0,1,0), (0,1,0,0), (1,0,0,0) are closest.
            var aPoint = 0x01
            var aScore = xins
            var bPoint = 0x02
            var bScore = yins
            if (aScore >= bScore && zins > bScore) {
                bScore = zins
                bPoint = 0x04
            }
            else if (aScore < bScore && zins > aScore) {
                aScore = zins
                aPoint = 0x04
            }
            if (aScore >= bScore && wins > bScore) {
                bScore = wins
                bPoint = 0x08
            }
            else if (aScore < bScore && wins > aScore) {
                aScore = wins
                aPoint = 0x08
            }

            //Now we determine the three lattice points not part of the pentachoron that may contribute.
            //This depends on the closest two pentachoron vertices, including (0,0,0,0)
            let uins = 1 - inSum
            if (uins > aScore || uins > bScore) { //(0,0,0,0) is one of the closest two pentachoron vertices.
                let c = UInt8(bScore > aScore ? bPoint : aPoint) //Our other closest vertex is the closest out of a and b.
                if ((c & 0x01) == 0) {
                    xsv_ext0 = xsb - 1
                    xsv_ext2 = xsb
                    xsv_ext1 = xsv_ext2
                    dx_ext0 = dx0 + 1
                    dx_ext2 = dx0
                    dx_ext1 = dx_ext2
                }
                else {
                    xsv_ext2 = xsb + 1
                    xsv_ext1 = xsv_ext2
                    xsv_ext0 = xsv_ext1
                    dx_ext2 = dx0 - 1
                    dx_ext1 = dx_ext2
                    dx_ext0 = dx_ext1
                }

                if ((c & 0x02) == 0) {
                    ysv_ext2 = ysb
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext2 = dy0
                    dy_ext1 = dy_ext2
                    dy_ext0 = dy_ext1
                    if ((c & 0x01) == 0x01) {
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                    }
                    else {
                        ysv_ext1 -= 1
                        dy_ext1 += 1
                    }
                }
                else {
                    ysv_ext2 = ysb + 1
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext2 = dy0 - 1
                    dy_ext1 = dy_ext2
                    dy_ext0 = dy_ext1
                }

                if ((c & 0x04) == 0) {
                    zsv_ext2 = zsb
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext2 = dz0
                    dz_ext1 = dz_ext2
                    dz_ext0 = dz_ext1
                    if ((c & 0x03) != 0) {
                        if ((c & 0x03) == 0x03) {
                            zsv_ext0 -= 1
                            dz_ext0 += 1
                        }
                        else {
                            zsv_ext1 -= 1
                            dz_ext1 += 1
                        }
                    }
                    else {
                        zsv_ext2 -= 1
                        dz_ext2 += 1
                    }
                }
                else {
                    zsv_ext2 = zsb + 1
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext2 = dz0 - 1
                    dz_ext1 = dz_ext2
                    dz_ext0 = dz_ext1
                }

                if ((c & 0x08) == 0) {
                    wsv_ext1 = wsb
                    wsv_ext0 = wsv_ext1
                    wsv_ext2 = wsb - 1
                    dw_ext1 = dw0
                    dw_ext0 = dw_ext1
                    dw_ext2 = dw0 + 1
                }
                else {
                    wsv_ext2 = wsb + 1
                    wsv_ext1 = wsv_ext2
                    wsv_ext0 = wsv_ext1
                    dw_ext2 = dw0 - 1
                    dw_ext1 = dw_ext2
                    dw_ext0 = dw_ext1
                }
            }
            else { //(0,0,0,0) is not one of the closest two pentachoron vertices.
                let c = UInt8(aPoint | bPoint) //Our three extra vertices are determined by the closest two.

                if ((c & 0x01) == 0) {
                    xsv_ext2 = xsb
                    xsv_ext0 = xsv_ext2
                    xsv_ext1 = xsb - 1
                    dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT_4D
                    dx_ext2 = dx0 - SQUISH_CONSTANT_4D
                }
                else {
                    xsv_ext2 = xsb + 1
                    xsv_ext1 = xsv_ext2
                    xsv_ext0 = xsv_ext1
                    dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dx_ext2 = dx0 - 1 - SQUISH_CONSTANT_4D
                    dx_ext1 = dx_ext2
                }

                if ((c & 0x02) == 0) {
                    ysv_ext2 = ysb
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2
                    if ((c & 0x01) == 0x01) {
                        ysv_ext1 -= 1
                        dy_ext1 += 1
                    }
                    else {
                        ysv_ext2 -= 1
                        dy_ext2 += 1
                    }
                }
                else {
                    ysv_ext2 = ysb + 1
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 1 - SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2
                }

                if ((c & 0x04) == 0) {
                    zsv_ext2 = zsb
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2
                    if ((c & 0x03) == 0x03) {
                        zsv_ext1 -= 1
                        dz_ext1 += 1
                    }
                    else {
                        zsv_ext2 -= 1
                        dz_ext2 += 1
                    }
                }
                else {
                    zsv_ext2 = zsb + 1
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 1 - SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2
                }

                if ((c & 0x08) == 0) {
                    wsv_ext1 = wsb
                    wsv_ext0 = wsv_ext1
                    wsv_ext2 = wsb - 1
                    dw_ext0 = dw0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 + 1 - SQUISH_CONSTANT_4D
                }
                else {
                    wsv_ext2 = wsb + 1
                    wsv_ext1 = wsv_ext2
                    wsv_ext0 = wsv_ext1
                    dw_ext0 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 1 - SQUISH_CONSTANT_4D
                    dw_ext1 = dw_ext2
                }
            }

            //Contribution (0,0,0,0)
            var attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0 - dw0 * dw0
            if (attn0 > 0) {
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 0, wsb: wsb + 0, dx: dx0, dy: dy0, dz: dz0, dw: dw0)
            }

            //Contribution (1,0,0,0)
            let dx1 = dx0 - 1 - SQUISH_CONSTANT_4D
            let dy1 = dy0 - 0 - SQUISH_CONSTANT_4D
            let dz1 = dz0 - 0 - SQUISH_CONSTANT_4D
            let dw1 = dw0 - 0 - SQUISH_CONSTANT_4D
            var attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if (attn1 > 0) {
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 0, wsb: wsb + 0, dx: dx1, dy: dy1, dz: dz1, dw: dw1)
            }

            //Contribution (0,1,0,0)
            let dx2 = dx0 - 0 - SQUISH_CONSTANT_4D
            let dy2 = dy0 - 1 - SQUISH_CONSTANT_4D
            let dz2 = dz1
            let dw2 = dw1
            var attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if (attn2 > 0) {
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 0, dx: dx2, dy: dy2, dz: dz2, dw: dw2)
            }

            //Contribution (0,0,1,0)
            let dx3 = dx2
            let dy3 = dy1
            let dz3 = dz0 - 1 - SQUISH_CONSTANT_4D
            let dw3 = dw1
            var attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if (attn3 > 0) {
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 0, dx: dx3, dy: dy3, dz: dz3, dw: dw3)
            }

            //Contribution (0,0,0,1)
            let dx4 = dx2
            let dy4 = dy1
            let dz4 = dz1
            let dw4 = dw0 - 1 - SQUISH_CONSTANT_4D
            var attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if (attn4 > 0) {
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 0, wsb: wsb + 1, dx: dx4, dy: dy4, dz: dz4, dw: dw4)
            }
        }
        else if (inSum >= 3) { //We're inside the pentachoron (4-Simplex) at (1,1,1,1)
            //Determine which two of (1,1,1,0), (1,1,0,1), (1,0,1,1), (0,1,1,1) are closest.
            var aPoint = 0x0E
            var aScore = xins
            var bPoint = 0x0D
            var bScore = yins
            if (aScore <= bScore && zins < bScore) {
                bScore = zins
                bPoint = 0x0B
            }
            else if (aScore > bScore && zins < aScore) {
                aScore = zins
                aPoint = 0x0B
            }
            if (aScore <= bScore && wins < bScore) {
                bScore = wins
                bPoint = 0x07
            }
            else if (aScore > bScore && wins < aScore) {
                aScore = wins
                aPoint = 0x07
            }

            //Now we determine the three lattice points not part of the pentachoron that may contribute.
            //This depends on the closest two pentachoron vertices, including (0,0,0,0)
            let uins = 4 - inSum
            if (uins < aScore || uins < bScore) { //(1,1,1,1) is one of the closest two pentachoron vertices.
                let c = (bScore < aScore ? bPoint : aPoint) //Our other closest vertex is the closest out of a and b.

                if ((c & 0x01) != 0) {
                    xsv_ext0 = xsb + 2
                    xsv_ext2 = xsb + 1
                    xsv_ext1 = xsv_ext2
                    dx_ext0 = dx0 - 2 - 4 * SQUISH_CONSTANT_4D
                    dx_ext2 = dx0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx_ext2
                }
                else {
                    xsv_ext2 = xsb
                    xsv_ext1 = xsv_ext2
                    xsv_ext0 = xsv_ext1
                    dx_ext2 = dx0 - 4 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx_ext2
                    dx_ext0 = dx_ext1
                }

                if ((c & 0x02) != 0) {
                    ysv_ext2 = ysb + 1
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext2 = dy0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2
                    dy_ext0 = dy_ext1
                    if ((c & 0x01) != 0) {
                        ysv_ext1 += 1
                        dy_ext1 -= 1
                    }
                    else {
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                    }
                }
                else {
                    ysv_ext2 = ysb
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext2 = dy0 - 4 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2
                    dy_ext0 = dy_ext1
                }

                if ((c & 0x04) != 0) {
                    zsv_ext2 = zsb + 1
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext2 = dz0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2
                    dz_ext0 = dz_ext1
                    if ((c & 0x03) != 0x03) {
                        if ((c & 0x03) == 0) {
                            zsv_ext0 += 1
                            dz_ext0 -= 1
                        }
                        else {
                            zsv_ext1 += 1
                            dz_ext1 -= 1
                        }
                    }
                    else {
                        zsv_ext2 += 1
                        dz_ext2 -= 1
                    }
                }
                else {
                    zsv_ext2 = zsb
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext2 = dz0 - 4 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2
                    dz_ext0 = dz_ext1
                }

                if ((c & 0x08) != 0) {
                    wsv_ext1 = wsb + 1
                    wsv_ext0 = wsv_ext1
                    wsv_ext2 = wsb + 2
                    dw_ext1 = dw0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dw_ext0 = dw_ext1
                    dw_ext2 = dw0 - 2 - 4 * SQUISH_CONSTANT_4D
                }
                else {
                    wsv_ext2 = wsb
                    wsv_ext1 = wsv_ext2
                    wsv_ext0 = wsv_ext1
                    dw_ext2 = dw0 - 4 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw_ext2
                    dw_ext0 = dw_ext1
                }
            }
            else { //(1,1,1,1) is not one of the closest two pentachoron vertices.
                let c = UInt8(aPoint & bPoint) //Our three extra vertices are determined by the closest two.

                if ((c & 0x01) != 0) {
                    xsv_ext2 = xsb + 1
                    xsv_ext0 = xsv_ext2
                    xsv_ext1 = xsb + 2
                    dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - 2 - 3 * SQUISH_CONSTANT_4D
                    dx_ext2 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                }
                else {
                    xsv_ext2 = xsb
                    xsv_ext1 = xsv_ext2
                    xsv_ext0 = xsv_ext1
                    dx_ext0 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dx_ext2 = dx0 - 3 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx_ext2
                }

                if ((c & 0x02) != 0) {
                    ysv_ext2 = ysb + 1
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2
                    if ((c & 0x01) != 0) {
                        ysv_ext2 += 1
                        dy_ext2 -= 1
                    }
                    else {
                        ysv_ext1 += 1
                        dy_ext1 -= 1
                    }
                }
                else {
                    ysv_ext2 = ysb
                    ysv_ext1 = ysv_ext2
                    ysv_ext0 = ysv_ext1
                    dy_ext0 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 3 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy_ext2
                }

                if ((c & 0x04) != 0) {
                    zsv_ext2 = zsb + 1
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2
                    if ((c & 0x03) != 0) {
                        zsv_ext2 += 1
                        dz_ext2 -= 1
                    }
                    else {
                        zsv_ext1 += 1
                        dz_ext1 -= 1
                    }
                }
                else {
                    zsv_ext2 = zsb
                    zsv_ext1 = zsv_ext2
                    zsv_ext0 = zsv_ext1
                    dz_ext0 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 3 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz_ext2
                }

                if ((c & 0x08) != 0) {
                    wsv_ext1 = wsb + 1
                    wsv_ext0 = wsv_ext1
                    wsv_ext2 = wsb + 2
                    dw_ext0 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 2 - 3 * SQUISH_CONSTANT_4D
                }
                else {
                    wsv_ext2 = wsb
                    wsv_ext1 = wsv_ext2
                    wsv_ext0 = wsv_ext1
                    dw_ext0 = dw0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 3 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw_ext2
                }
            }

            //Contribution (1,1,1,0)
            let dx4 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
            let dy4 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
            let dz4 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
            let dw4 = dw0 - 3 * SQUISH_CONSTANT_4D
            var attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if (attn4 > 0) {
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 1, wsb: wsb + 0, dx: dx4, dy: dy4, dz: dz4, dw: dw4)
            }

            //Contribution (1,1,0,1)
            let dx3 = dx4
            let dy3 = dy4
            let dz3 = dz0 - 3 * SQUISH_CONSTANT_4D
            let dw3 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
            var attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if (attn3 > 0) {
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 1, dx: dx3, dy: dy3, dz: dz3, dw: dw3)
            }

            //Contribution (1,0,1,1)
            let dx2 = dx4
            let dy2 = dy0 - 3 * SQUISH_CONSTANT_4D
            let dz2 = dz4
            let dw2 = dw3
            var attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if (attn2 > 0) {
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 1, dx: dx2, dy: dy2, dz: dz2, dw: dw2)
            }

            //Contribution (0,1,1,1)
            let dx1 = dx0 - 3 * SQUISH_CONSTANT_4D
            let dz1 = dz4
            let dy1 = dy4
            let dw1 = dw3
            var attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if (attn1 > 0) {
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 1, wsb: wsb + 1, dx: dx1, dy: dy1, dz: dz1, dw: dw1)
            }

            //Contribution (1,1,1,1)
            dx0 = dx0 - 1 - 4 * SQUISH_CONSTANT_4D
            dy0 = dy0 - 1 - 4 * SQUISH_CONSTANT_4D
            dz0 = dz0 - 1 - 4 * SQUISH_CONSTANT_4D
            dw0 = dw0 - 1 - 4 * SQUISH_CONSTANT_4D
            var attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0 - dw0 * dw0
            if (attn0 > 0) {
                attn0 *= attn0
                value += attn0 * attn0 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 1, wsb: wsb + 1, dx: dx0, dy: dy0, dz: dz0, dw: dw0)
            }
        }
        else if (inSum <= 2) { //We're inside the first dispentachoron (Rectified 4-Simplex)
            var aScore: Double
            var aPoint: UInt8
            var aIsBiggerSide = true
            var bScore: Double
            var bPoint: UInt8
            var bIsBiggerSide = true

            //Decide between (1,1,0,0) and (0,0,1,1)
            if (xins + yins > zins + wins) {
                aScore = xins + yins
                aPoint = 0x03
            }
            else {
                aScore = zins + wins
                aPoint = 0x0C
            }

            //Decide between (1,0,1,0) and (0,1,0,1)
            if (xins + zins > yins + wins) {
                bScore = xins + zins
                bPoint = 0x05
            }
            else {
                bScore = yins + wins
                bPoint = 0x0A
            }

            //Closer between (1,0,0,1) and (0,1,1,0) will replace the further of a and b, if closer.
            if (xins + wins > yins + zins) {
                let score = xins + wins
                if (aScore >= bScore && score > bScore) {
                    bScore = score
                    bPoint = 0x09
                }
                else if (aScore < bScore && score > aScore) {
                    aScore = score
                    aPoint = 0x09
                }
            }
            else {
                let score = yins + zins
                if (aScore >= bScore && score > bScore) {
                    bScore = score
                    bPoint = 0x06
                }
                else if (aScore < bScore && score > aScore) {
                    aScore = score
                    aPoint = 0x06
                }
            }

            //Decide if (1,0,0,0) is closer.
            let p1 = 2 - inSum + xins
            if (aScore >= bScore && p1 > bScore) {
                bScore = p1
                bPoint = 0x01
                bIsBiggerSide = false
            }
            else if (aScore < bScore && p1 > aScore) {
                aScore = p1
                aPoint = 0x01
                aIsBiggerSide = false
            }

            //Decide if (0,1,0,0) is closer.
            let p2 = 2 - inSum + yins
            if (aScore >= bScore && p2 > bScore) {
                bScore = p2
                bPoint = 0x02
                bIsBiggerSide = false
            }
            else if (aScore < bScore && p2 > aScore) {
                aScore = p2
                aPoint = 0x02
                aIsBiggerSide = false
            }

            //Decide if (0,0,1,0) is closer.
            let p3 = 2 - inSum + zins
            if (aScore >= bScore && p3 > bScore) {
                bScore = p3
                bPoint = 0x04
                bIsBiggerSide = false
            }
            else if (aScore < bScore && p3 > aScore) {
                aScore = p3
                aPoint = 0x04
                aIsBiggerSide = false
            }

            //Decide if (0,0,0,1) is closer.
            let p4 = 2 - inSum + wins
            if (aScore >= bScore && p4 > bScore) {
                bScore = p4
                bPoint = 0x08
                bIsBiggerSide = false
            }
            else if (aScore < bScore && p4 > aScore) {
                aScore = p4
                aPoint = 0x08
                aIsBiggerSide = false
            }

            //Where each of the two closest points are determines how the extra three vertices are calculated.
            if (aIsBiggerSide == bIsBiggerSide) {
                if (aIsBiggerSide) { //Both closest points on the bigger side
                    let c1 = UInt8(aPoint | bPoint)
                    let c2 = UInt8(aPoint & bPoint)
                    if ((c1 & 0x01) == 0) {
                        xsv_ext0 = xsb
                        xsv_ext1 = xsb - 1
                        dx_ext0 = dx0 - 3 * SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 + 1 - 2 * SQUISH_CONSTANT_4D
                    }
                    else {
                        xsv_ext1 = xsb + 1
                        xsv_ext0 = xsv_ext1
                        dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                    }

                    if ((c1 & 0x02) == 0) {
                        ysv_ext0 = ysb
                        ysv_ext1 = ysb - 1
                        dy_ext0 = dy0 - 3 * SQUISH_CONSTANT_4D
                        dy_ext1 = dy0 + 1 - 2 * SQUISH_CONSTANT_4D
                    }
                    else {
                        ysv_ext1 = ysb + 1
                        ysv_ext0 = ysv_ext1
                        dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dy_ext1 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                    }

                    if ((c1 & 0x04) == 0) {
                        zsv_ext0 = zsb
                        zsv_ext1 = zsb - 1
                        dz_ext0 = dz0 - 3 * SQUISH_CONSTANT_4D
                        dz_ext1 = dz0 + 1 - 2 * SQUISH_CONSTANT_4D
                    }
                    else {
                        zsv_ext1 = zsb + 1
                        zsv_ext0 = zsv_ext1
                        dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dz_ext1 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                    }

                    if ((c1 & 0x08) == 0) {
                        wsv_ext0 = wsb
                        wsv_ext1 = wsb - 1
                        dw_ext0 = dw0 - 3 * SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 + 1 - 2 * SQUISH_CONSTANT_4D
                    }
                    else {
                        wsv_ext1 = wsb + 1
                        wsv_ext0 = wsv_ext1
                        dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                    }

                    //One combination is a permutation of (0,0,0,2) based on c2
                    xsv_ext2 = xsb
                    ysv_ext2 = ysb
                    zsv_ext2 = zsb
                    wsv_ext2 = wsb
                    dx_ext2 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 2 * SQUISH_CONSTANT_4D
                    if ((c2 & 0x01) != 0) {
                        xsv_ext2 += 2
                        dx_ext2 -= 2
                    }
                    else if ((c2 & 0x02) != 0) {
                        ysv_ext2 += 2
                        dy_ext2 -= 2
                    }
                    else if ((c2 & 0x04) != 0) {
                        zsv_ext2 += 2
                        dz_ext2 -= 2
                    }
                    else {
                        wsv_ext2 += 2
                        dw_ext2 -= 2
                    }

                }
                else { //Both closest points on the smaller side
                    //One of the two extra points is (0,0,0,0)
                    xsv_ext2 = xsb
                    ysv_ext2 = ysb
                    zsv_ext2 = zsb
                    wsv_ext2 = wsb
                    dx_ext2 = dx0
                    dy_ext2 = dy0
                    dz_ext2 = dz0
                    dw_ext2 = dw0

                    //Other two points are based on the omitted axes.
                    let c = UInt8(aPoint | bPoint)

                    if ((c & 0x01) == 0) {
                        xsv_ext0 = xsb - 1
                        xsv_ext1 = xsb
                        dx_ext0 = dx0 + 1 - SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 - SQUISH_CONSTANT_4D
                    }
                    else {
                        xsv_ext1 = xsb + 1
                        xsv_ext0 = xsv_ext1
                        dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_4D
                        dx_ext0 = dx_ext1
                    }

                    if ((c & 0x02) == 0) {
                        ysv_ext1 = ysb
                        ysv_ext0 = ysv_ext1
                        dy_ext1 = dy0 - SQUISH_CONSTANT_4D
                        dy_ext0 = dy_ext1
                        if ((c & 0x01) == 0x01) {
                            ysv_ext0 -= 1
                            dy_ext0 += 1
                        }
                        else {
                            ysv_ext1 -= 1
                            dy_ext1 += 1
                        }
                    }
                    else {
                        ysv_ext1 = ysb + 1
                        ysv_ext0 = ysv_ext1
                        dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_4D
                        dy_ext0 = dy_ext1
                    }

                    if ((c & 0x04) == 0) {
                        zsv_ext1 = zsb
                        zsv_ext0 = zsv_ext1
                        dz_ext1 = dz0 - SQUISH_CONSTANT_4D
                        dz_ext0 = dz_ext1
                        if ((c & 0x03) == 0x03) {
                            zsv_ext0 -= 1
                            dz_ext0 += 1
                        }
                        else {
                            zsv_ext1 -= 1
                            dz_ext1 += 1
                        }
                    }
                    else {
                        zsv_ext1 = zsb + 1
                        zsv_ext0 = zsv_ext1
                        dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_4D
                        dz_ext0 = dz_ext1
                    }

                    if ((c & 0x08) == 0) {
                        wsv_ext0 = wsb
                        wsv_ext1 = wsb - 1
                        dw_ext0 = dw0 - SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 + 1 - SQUISH_CONSTANT_4D
                    }
                    else {
                        wsv_ext1 = wsb + 1
                        wsv_ext0 = wsv_ext1
                        dw_ext1 = dw0 - 1 - SQUISH_CONSTANT_4D
                        dw_ext0 = dw_ext1
                    }

                }
            }
            else { //One point on each "side"
                var c1: UInt8, c2: UInt8
                if (aIsBiggerSide) {
                    c1 = aPoint
                    c2 = bPoint
                }
                else {
                    c1 = bPoint
                    c2 = aPoint
                }

                //Two contributions are the bigger-sided point with each 0 replaced with -1.
                if ((c1 & 0x01) == 0) {
                    xsv_ext0 = xsb - 1
                    xsv_ext1 = xsb
                    dx_ext0 = dx0 + 1 - SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - SQUISH_CONSTANT_4D
                }
                else {
                    xsv_ext1 = xsb + 1
                    xsv_ext0 = xsv_ext1
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT_4D
                    dx_ext0 = dx_ext1
                }

                if ((c1 & 0x02) == 0) {
                    ysv_ext1 = ysb
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0 - SQUISH_CONSTANT_4D
                    dy_ext0 = dy_ext1
                    if ((c1 & 0x01) == 0x01) {
                        ysv_ext0 -= 1
                        dy_ext0 += 1
                    }
                    else {
                        ysv_ext1 -= 1
                        dy_ext1 += 1
                    }
                }
                else {
                    ysv_ext1 = ysb + 1
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT_4D
                    dy_ext0 = dy_ext1
                }

                if ((c1 & 0x04) == 0) {
                    zsv_ext1 = zsb
                    zsv_ext0 = zsv_ext1
                    dz_ext1 = dz0 - SQUISH_CONSTANT_4D
                    dz_ext0 = dz_ext1
                    if ((c1 & 0x03) == 0x03) {
                        zsv_ext0 -= 1
                        dz_ext0 += 1
                    }
                    else {
                        zsv_ext1 -= 1
                        dz_ext1 += 1
                    }
                }
                else {
                    zsv_ext1 = zsb + 1
                    zsv_ext0 = zsv_ext1
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT_4D
                    dz_ext0 = dz_ext1
                }

                if ((c1 & 0x08) == 0) {
                    wsv_ext0 = wsb
                    wsv_ext1 = wsb - 1
                    dw_ext0 = dw0 - SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 + 1 - SQUISH_CONSTANT_4D
                }
                else {
                    wsv_ext1 = wsb + 1
                    wsv_ext0 = wsv_ext1
                    dw_ext1 = dw0 - 1 - SQUISH_CONSTANT_4D
                    dw_ext0 = dw_ext1
                }

                //One contribution is a permutation of (0,0,0,2) based on the smaller-sided point
                xsv_ext2 = xsb
                ysv_ext2 = ysb
                zsv_ext2 = zsb
                wsv_ext2 = wsb
                dx_ext2 = dx0 - 2 * SQUISH_CONSTANT_4D
                dy_ext2 = dy0 - 2 * SQUISH_CONSTANT_4D
                dz_ext2 = dz0 - 2 * SQUISH_CONSTANT_4D
                dw_ext2 = dw0 - 2 * SQUISH_CONSTANT_4D
                if ((c2 & 0x01) != 0) {
                    xsv_ext2 += 2
                    dx_ext2 -= 2
                }
                else if ((c2 & 0x02) != 0) {
                    ysv_ext2 += 2
                    dy_ext2 -= 2
                }
                else if ((c2 & 0x04) != 0) {
                    zsv_ext2 += 2
                    dz_ext2 -= 2
                }
                else {
                    wsv_ext2 += 2
                    dw_ext2 -= 2
                }
            }

            //Contribution (1,0,0,0)
            let dx1 = dx0 - 1 - SQUISH_CONSTANT_4D
            let dy1 = dy0 - 0 - SQUISH_CONSTANT_4D
            let dz1 = dz0 - 0 - SQUISH_CONSTANT_4D
            let dw1 = dw0 - 0 - SQUISH_CONSTANT_4D
            var attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if (attn1 > 0) {
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 0, wsb: wsb + 0, dx: dx1, dy: dy1, dz: dz1, dw: dw1)
            }

            //Contribution (0,1,0,0)
            let dx2 = dx0 - 0 - SQUISH_CONSTANT_4D
            let dy2 = dy0 - 1 - SQUISH_CONSTANT_4D
            let dz2 = dz1
            let dw2 = dw1
            var attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if (attn2 > 0) {
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 0, dx: dx2, dy: dy2, dz: dz2, dw: dw2)
            }

            //Contribution (0,0,1,0)
            let dx3 = dx2
            let dy3 = dy1
            let dz3 = dz0 - 1 - SQUISH_CONSTANT_4D
            let dw3 = dw1
            var attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if (attn3 > 0) {
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 0, dx: dx3, dy: dy3, dz: dz3, dw: dw3)
            }

            //Contribution (0,0,0,1)
            let dx4 = dx2
            let dy4 = dy1
            let dz4 = dz1
            let dw4 = dw0 - 1 - SQUISH_CONSTANT_4D
            var attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if (attn4 > 0) {
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 0, wsb: wsb + 1, dx: dx4, dy: dy4, dz: dz4, dw: dw4)
            }

            //Contribution (1,1,0,0)
            let dx5 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dy5 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dz5 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dw5 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            var attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5 - dw5 * dw5
            if (attn5 > 0) {
                attn5 *= attn5
                value += attn5 * attn5 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 0, dx: dx5, dy: dy5, dz: dz5, dw: dw5)
            }

            //Contribution (1,0,1,0)
            let dx6 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dy6 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dz6 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dw6 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            var attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6 - dw6 * dw6
            if (attn6 > 0) {
                attn6 *= attn6
                value += attn6 * attn6 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 0, dx: dx6, dy: dy6, dz: dz6, dw: dw6)
            }

            //Contribution (1,0,0,1)
            let dx7 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dy7 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dz7 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dw7 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            var attn7 = 2 - dx7 * dx7 - dy7 * dy7 - dz7 * dz7 - dw7 * dw7
            if (attn7 > 0) {
                attn7 *= attn7
                value += attn7 * attn7 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 0, wsb: wsb + 1, dx: dx7, dy: dy7, dz: dz7, dw: dw7)
            }

            //Contribution (0,1,1,0)
            let dx8 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dy8 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dz8 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dw8 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            var attn8 = 2 - dx8 * dx8 - dy8 * dy8 - dz8 * dz8 - dw8 * dw8
            if (attn8 > 0) {
                attn8 *= attn8
                value += attn8 * attn8 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 1, wsb: wsb + 0, dx: dx8, dy: dy8, dz: dz8, dw: dw8)
            }

            //Contribution (0,1,0,1)
            let dx9 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dy9 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dz9 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dw9 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            var attn9 = 2 - dx9 * dx9 - dy9 * dy9 - dz9 * dz9 - dw9 * dw9
            if (attn9 > 0) {
                attn9 *= attn9
                value += attn9 * attn9 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 1, dx: dx9, dy: dy9, dz: dz9, dw: dw9)
            }

            //Contribution (0,0,1,1)
            let dx10 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dy10 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dz10 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dw10 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            var attn10 = 2 - dx10 * dx10 - dy10 * dy10 - dz10 * dz10 - dw10 * dw10
            if (attn10 > 0) {
                attn10 *= attn10
                value += attn10 * attn10 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 1, dx: dx10, dy: dy10, dz: dz10, dw: dw10)
            }
        }
        else { //We're inside the second dispentachoron (Rectified 4-Simplex)
            var aScore: Double
            var aPoint: UInt8
            var aIsBiggerSide = true
            var bScore: Double
            var bPoint: UInt8
            var bIsBiggerSide = true

            //Decide between (0,0,1,1) and (1,1,0,0)
            if (xins + yins < zins + wins) {
                aScore = xins + yins
                aPoint = 0x0C
            }
            else {
                aScore = zins + wins
                aPoint = 0x03
            }

            //Decide between (0,1,0,1) and (1,0,1,0)
            if (xins + zins < yins + wins) {
                bScore = xins + zins
                bPoint = 0x0A
            }
            else {
                bScore = yins + wins
                bPoint = 0x05
            }

            //Closer between (0,1,1,0) and (1,0,0,1) will replace the further of a and b, if closer.
            if (xins + wins < yins + zins) {
                let score = xins + wins
                if (aScore <= bScore && score < bScore) {
                    bScore = score
                    bPoint = 0x06
                }
                else if (aScore > bScore && score < aScore) {
                    aScore = score
                    aPoint = 0x06
                }
            }
            else {
                let score = yins + zins
                if (aScore <= bScore && score < bScore) {
                    bScore = score
                    bPoint = 0x09
                }
                else if (aScore > bScore && score < aScore) {
                    aScore = score
                    aPoint = 0x09
                }
            }

            //Decide if (0,1,1,1) is closer.
            let p1 = 3 - inSum + xins
            if (aScore <= bScore && p1 < bScore) {
                bScore = p1
                bPoint = 0x0E
                bIsBiggerSide = false
            }
            else if (aScore > bScore && p1 < aScore) {
                aScore = p1
                aPoint = 0x0E
                aIsBiggerSide = false
            }

            //Decide if (1,0,1,1) is closer.
            let p2 = 3 - inSum + yins
            if (aScore <= bScore && p2 < bScore) {
                bScore = p2
                bPoint = 0x0D
                bIsBiggerSide = false
            }
            else if (aScore > bScore && p2 < aScore) {
                aScore = p2
                aPoint = 0x0D
                aIsBiggerSide = false
            }

            //Decide if (1,1,0,1) is closer.
            let p3 = 3 - inSum + zins
            if (aScore <= bScore && p3 < bScore) {
                bScore = p3
                bPoint = 0x0B
                bIsBiggerSide = false
            }
            else if (aScore > bScore && p3 < aScore) {
                aScore = p3
                aPoint = 0x0B
                aIsBiggerSide = false
            }

            //Decide if (1,1,1,0) is closer.
            let p4 = 3 - inSum + wins
            if (aScore <= bScore && p4 < bScore) {
                bScore = p4
                bPoint = 0x07
                bIsBiggerSide = false
            }
            else if (aScore > bScore && p4 < aScore) {
                aScore = p4
                aPoint = 0x07
                aIsBiggerSide = false
            }

            //Where each of the two closest points are determines how the extra three vertices are calculated.
            if (aIsBiggerSide == bIsBiggerSide) {
                if (aIsBiggerSide) { //Both closest points on the bigger side
                    let c1 = UInt8(aPoint & bPoint)
                    let c2 = UInt8(aPoint | bPoint)

                    //Two contributions are permutations of (0,0,0,1) and (0,0,0,2) based on c1
                    xsv_ext1 = xsb
                    xsv_ext0 = xsv_ext1
                    ysv_ext1 = ysb
                    ysv_ext0 = ysv_ext1
                    zsv_ext1 = zsb
                    zsv_ext0 = zsv_ext1
                    wsv_ext1 = wsb
                    wsv_ext0 = wsv_ext1
                    dx_ext0 = dx0 - SQUISH_CONSTANT_4D
                    dy_ext0 = dy0 - SQUISH_CONSTANT_4D
                    dz_ext0 = dz0 - SQUISH_CONSTANT_4D
                    dw_ext0 = dw0 - SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT_4D
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT_4D
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - 2 * SQUISH_CONSTANT_4D
                    if ((c1 & 0x01) != 0) {
                        xsv_ext0 += 1
                        dx_ext0 -= 1
                        xsv_ext1 += 2
                        dx_ext1 -= 2
                    }
                    else if ((c1 & 0x02) != 0) {
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                        ysv_ext1 += 2
                        dy_ext1 -= 2
                    }
                    else if ((c1 & 0x04) != 0) {
                        zsv_ext0 += 1
                        dz_ext0 -= 1
                        zsv_ext1 += 2
                        dz_ext1 -= 2
                    }
                    else {
                        wsv_ext0 += 1
                        dw_ext0 -= 1
                        wsv_ext1 += 2
                        dw_ext1 -= 2
                    }

                    //One contribution is a permutation of (1,1,1,-1) based on c2
                    xsv_ext2 = xsb + 1
                    ysv_ext2 = ysb + 1
                    zsv_ext2 = zsb + 1
                    wsv_ext2 = wsb + 1
                    dx_ext2 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                    if ((c2 & 0x01) == 0) {
                        xsv_ext2 -= 2
                        dx_ext2 += 2
                    }
                    else if ((c2 & 0x02) == 0) {
                        ysv_ext2 -= 2
                        dy_ext2 += 2
                    }
                    else if ((c2 & 0x04) == 0) {
                        zsv_ext2 -= 2
                        dz_ext2 += 2
                    }
                    else {
                        wsv_ext2 -= 2
                        dw_ext2 += 2
                    }
                }
                else { //Both closest points on the smaller side
                    //One of the two extra points is (1,1,1,1)
                    xsv_ext2 = xsb + 1
                    ysv_ext2 = ysb + 1
                    zsv_ext2 = zsb + 1
                    wsv_ext2 = wsb + 1
                    dx_ext2 = dx0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dy_ext2 = dy0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dz_ext2 = dz0 - 1 - 4 * SQUISH_CONSTANT_4D
                    dw_ext2 = dw0 - 1 - 4 * SQUISH_CONSTANT_4D

                    //Other two points are based on the shared axes.
                    let c = UInt8(aPoint & bPoint)

                    if ((c & 0x01) != 0) {
                        xsv_ext0 = xsb + 2
                        xsv_ext1 = xsb + 1
                        dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT_4D
                        dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                    }
                    else {
                        xsv_ext1 = xsb
                        xsv_ext0 = xsv_ext1
                        dx_ext1 = dx0 - 3 * SQUISH_CONSTANT_4D
                        dx_ext0 = dx_ext1
                    }

                    if ((c & 0x02) != 0) {
                        ysv_ext1 = ysb + 1
                        ysv_ext0 = ysv_ext1
                        dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dy_ext0 = dy_ext1
                        if ((c & 0x01) == 0) {
                            ysv_ext0 += 1
                            dy_ext0 -= 1
                        }
                        else {
                            ysv_ext1 += 1
                            dy_ext1 -= 1
                        }
                    }
                    else {
                        ysv_ext1 = ysb
                        ysv_ext0 = ysv_ext1
                        dy_ext1 = dy0 - 3 * SQUISH_CONSTANT_4D
                        dy_ext0 = dy_ext1
                    }

                    if ((c & 0x04) != 0) {
                        zsv_ext1 = zsb + 1
                        zsv_ext0 = zsv_ext1
                        dz_ext1 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dz_ext0 = dz_ext1
                        if ((c & 0x03) == 0) {
                            zsv_ext0 += 1
                            dz_ext0 -= 1
                        }
                        else {
                            zsv_ext1 += 1
                            dz_ext1 -= 1
                        }
                    }
                    else {
                        zsv_ext1 = zsb
                        zsv_ext0 = zsv_ext1
                        dz_ext1 = dz0 - 3 * SQUISH_CONSTANT_4D
                        dz_ext0 = dz_ext1
                    }

                    if ((c & 0x08) != 0) {
                        wsv_ext0 = wsb + 1
                        wsv_ext1 = wsb + 2
                        dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                        dw_ext1 = dw0 - 2 - 3 * SQUISH_CONSTANT_4D
                    }
                    else {
                        wsv_ext1 = wsb
                        wsv_ext0 = wsv_ext1
                        dw_ext1 = dw0 - 3 * SQUISH_CONSTANT_4D
                        dw_ext0 = dw_ext1
                    }
                }
            }
            else { //One point on each "side"
                var c1: UInt8, c2: UInt8
                if (aIsBiggerSide) {
                    c1 = aPoint
                    c2 = bPoint
                }
                else {
                    c1 = bPoint
                    c2 = aPoint
                }

                //Two contributions are the bigger-sided point with each 1 replaced with 2.
                if ((c1 & 0x01) != 0) {
                    xsv_ext0 = xsb + 2
                    xsv_ext1 = xsb + 1
                    dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT_4D
                    dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
                }
                else {
                    xsv_ext1 = xsb
                    xsv_ext0 = xsv_ext1
                    dx_ext1 = dx0 - 3 * SQUISH_CONSTANT_4D
                    dx_ext0 = dx_ext1
                }

                if ((c1 & 0x02) != 0) {
                    ysv_ext1 = ysb + 1
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dy_ext0 = dy_ext1
                    if ((c1 & 0x01) == 0) {
                        ysv_ext0 += 1
                        dy_ext0 -= 1
                    }
                    else {
                        ysv_ext1 += 1
                        dy_ext1 -= 1
                    }
                }
                else {
                    ysv_ext1 = ysb
                    ysv_ext0 = ysv_ext1
                    dy_ext1 = dy0 - 3 * SQUISH_CONSTANT_4D
                    dy_ext0 = dy_ext1
                }

                if ((c1 & 0x04) != 0) {
                    zsv_ext1 = zsb + 1
                    zsv_ext0 = zsv_ext1
                    dz_ext1 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dz_ext0 = dz_ext1
                    if ((c1 & 0x03) == 0) {
                        zsv_ext0 += 1
                        dz_ext0 -= 1
                    }
                    else {
                        zsv_ext1 += 1
                        dz_ext1 -= 1
                    }
                }
                else {
                    zsv_ext1 = zsb
                    zsv_ext0 = zsv_ext1
                    dz_ext1 = dz0 - 3 * SQUISH_CONSTANT_4D
                    dz_ext0 = dz_ext1
                }

                if ((c1 & 0x08) != 0) {
                    wsv_ext0 = wsb + 1
                    wsv_ext1 = wsb + 2
                    dw_ext0 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
                    dw_ext1 = dw0 - 2 - 3 * SQUISH_CONSTANT_4D
                }
                else {
                    wsv_ext1 = wsb
                    wsv_ext0 = wsv_ext1
                    dw_ext1 = dw0 - 3 * SQUISH_CONSTANT_4D
                    dw_ext0 = dw_ext1
                }

                //One contribution is a permutation of (1,1,1,-1) based on the smaller-sided point
                xsv_ext2 = xsb + 1
                ysv_ext2 = ysb + 1
                zsv_ext2 = zsb + 1
                wsv_ext2 = wsb + 1
                dx_ext2 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
                dy_ext2 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
                dz_ext2 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
                dw_ext2 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
                if ((c2 & 0x01) == 0) {
                    xsv_ext2 -= 2
                    dx_ext2 += 2
                }
                else if ((c2 & 0x02) == 0) {
                    ysv_ext2 -= 2
                    dy_ext2 += 2
                }
                else if ((c2 & 0x04) == 0) {
                    zsv_ext2 -= 2
                    dz_ext2 += 2
                }
                else {
                    wsv_ext2 -= 2
                    dw_ext2 += 2
                }
            }

            //Contribution (1,1,1,0)
            let dx4 = dx0 - 1 - 3 * SQUISH_CONSTANT_4D
            let dy4 = dy0 - 1 - 3 * SQUISH_CONSTANT_4D
            let dz4 = dz0 - 1 - 3 * SQUISH_CONSTANT_4D
            let dw4 = dw0 - 3 * SQUISH_CONSTANT_4D
            var attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4 - dw4 * dw4
            if (attn4 > 0) {
                attn4 *= attn4
                value += attn4 * attn4 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 1, wsb: wsb + 0, dx: dx4, dy: dy4, dz: dz4, dw: dw4)
            }

            //Contribution (1,1,0,1)
            let dx3 = dx4
            let dy3 = dy4
            let dz3 = dz0 - 3 * SQUISH_CONSTANT_4D
            let dw3 = dw0 - 1 - 3 * SQUISH_CONSTANT_4D
            var attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3 - dw3 * dw3
            if (attn3 > 0) {
                attn3 *= attn3
                value += attn3 * attn3 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 1, dx: dx3, dy: dy3, dz: dz3, dw: dw3)
            }

            //Contribution (1,0,1,1)
            let dx2 = dx4
            let dy2 = dy0 - 3 * SQUISH_CONSTANT_4D
            let dz2 = dz4
            let dw2 = dw3
            var attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2 - dw2 * dw2
            if (attn2 > 0) {
                attn2 *= attn2
                value += attn2 * attn2 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 1, dx: dx2, dy: dy2, dz: dz2, dw: dw2)
            }

            //Contribution (0,1,1,1)
            let dx1 = dx0 - 3 * SQUISH_CONSTANT_4D
            let dz1 = dz4
            let dy1 = dy4
            let dw1 = dw3
            var attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1 - dw1 * dw1
            if (attn1 > 0) {
                attn1 *= attn1
                value += attn1 * attn1 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 1, wsb: wsb + 1, dx: dx1, dy: dy1, dz: dz1, dw: dw1)
            }

            //Contribution (1,1,0,0)
            let dx5 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dy5 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dz5 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dw5 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            var attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5 - dw5 * dw5
            if (attn5 > 0) {
                attn5 *= attn5
                value += attn5 * attn5 * extrapolate(xsb: xsb + 1, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 0, dx: dx5, dy: dy5, dz: dz5, dw: dw5)
            }

            //Contribution (1,0,1,0)
            let dx6 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dy6 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dz6 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dw6 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            var attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6 - dw6 * dw6
            if (attn6 > 0) {
                attn6 *= attn6
                value += attn6 * attn6 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 0, dx: dx6, dy: dy6, dz: dz6, dw: dw6)
            }

            //Contribution (1,0,0,1)
            let dx7 = dx0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dy7 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dz7 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dw7 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            var attn7 = 2 - dx7 * dx7 - dy7 * dy7 - dz7 * dz7 - dw7 * dw7
            if (attn7 > 0) {
                attn7 *= attn7
                value += attn7 * attn7 * extrapolate(xsb: xsb + 1, ysb: ysb + 0, zsb: zsb + 0, wsb: wsb + 1, dx: dx7, dy: dy7, dz: dz7, dw: dw7)
            }

            //Contribution (0,1,1,0)
            let dx8 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dy8 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dz8 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dw8 = dw0 - 0 - 2 * SQUISH_CONSTANT_4D
            var attn8 = 2 - dx8 * dx8 - dy8 * dy8 - dz8 * dz8 - dw8 * dw8
            if (attn8 > 0) {
                attn8 *= attn8
                value += attn8 * attn8 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 1, wsb: wsb + 0, dx: dx8, dy: dy8, dz: dz8, dw: dw8)
            }

            //Contribution (0,1,0,1)
            let dx9 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dy9 = dy0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dz9 = dz0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dw9 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            var attn9 = 2 - dx9 * dx9 - dy9 * dy9 - dz9 * dz9 - dw9 * dw9
            if (attn9 > 0) {
                attn9 *= attn9
                value += attn9 * attn9 * extrapolate(xsb: xsb + 0, ysb: ysb + 1, zsb: zsb + 0, wsb: wsb + 1, dx: dx9, dy: dy9, dz: dz9, dw: dw9)
            }

            //Contribution (0,0,1,1)
            let dx10 = dx0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dy10 = dy0 - 0 - 2 * SQUISH_CONSTANT_4D
            let dz10 = dz0 - 1 - 2 * SQUISH_CONSTANT_4D
            let dw10 = dw0 - 1 - 2 * SQUISH_CONSTANT_4D
            var attn10 = 2 - dx10 * dx10 - dy10 * dy10 - dz10 * dz10 - dw10 * dw10
            if (attn10 > 0) {
                attn10 *= attn10
                value += attn10 * attn10 * extrapolate(xsb: xsb + 0, ysb: ysb + 0, zsb: zsb + 1, wsb: wsb + 1, dx: dx10, dy: dy10, dz: dz10, dw: dw10)
            }
        }

        //First extra vertex
        var attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0 - dw_ext0 * dw_ext0
        if (attn_ext0 > 0) {
            attn_ext0 *= attn_ext0
            value += attn_ext0 * attn_ext0 * extrapolate(xsb: xsv_ext0, ysb: ysv_ext0, zsb: zsv_ext0, wsb: wsv_ext0, dx: dx_ext0, dy: dy_ext0, dz: dz_ext0, dw: dw_ext0)
        }

        //Second extra vertex
        var attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1 - dw_ext1 * dw_ext1
        if (attn_ext1 > 0) {
            attn_ext1 *= attn_ext1
            value += attn_ext1 * attn_ext1 * extrapolate(xsb: xsv_ext1, ysb: ysv_ext1, zsb: zsv_ext1, wsb: wsv_ext1, dx: dx_ext1, dy: dy_ext1, dz: dz_ext1, dw: dw_ext1)
        }

        //Third extra vertex
        var attn_ext2 = 2 - dx_ext2 * dx_ext2 - dy_ext2 * dy_ext2 - dz_ext2 * dz_ext2 - dw_ext2 * dw_ext2
        if (attn_ext2 > 0) {
            attn_ext2 *= attn_ext2
            value += attn_ext2 * attn_ext2 * extrapolate(xsb: xsv_ext2, ysb: ysv_ext2, zsb: zsv_ext2, wsb: wsv_ext2, dx: dx_ext2, dy: dy_ext2, dz: dz_ext2, dw: dw_ext2)
        }

        return value / Double(NORM_CONSTANT_4D)
    }

    private func extrapolate(xsb xsb: Int, ysb: Int, dx: Double, dy: Double) -> Double {
        let index = Int(perm[Int((perm[xsb & 0xFF] + ysb) & 0xFF)] & 0x0E)
        return Double(gradients2D[index]) * dx
        + Double(gradients2D[index + 1]) * dy
    }

    private func extrapolate(xsb xsb: Int, ysb: Int, zsb: Int, dx: Double, dy: Double, dz: Double) -> Double {
        let index = Int(permGradIndex3D[Int((perm[Int((perm[xsb & 0xFF] + ysb) & 0xFF)] + zsb) & 0xFF)])
        return Double(gradients3D[index]) * dx
        + Double(gradients3D[index + 1]) * dy
        + Double(gradients3D[index + 2]) * dz
    }

    private func extrapolate(xsb xsb: Int, ysb: Int, zsb: Int, wsb: Int, dx: Double, dy: Double, dz: Double, dw: Double) -> Double {
        let index = Int(perm[Int((perm[Int((perm[Int((perm[Int(xsb & 0xFF)] + ysb) & 0xFF)] + zsb) & 0xFF)] + wsb) & 0xFF)] & 0xFC)
        let ret = Double(gradients4D[index]) * dx
                    + Double(gradients4D[index + 1]) * dy
        return ret
        + Double(gradients4D[index + 2]) * dz
        + Double(gradients4D[index + 3]) * dw
    }

    private func fastFloor(x: Double) -> Int {
        let xi = Int(x)
        return x < Double(xi) ? xi - 1 : xi
    }

    //Gradients for 2D. They approximate the directions to the
    //vertices of an octagon from the center
    let gradients2D: [Int8] = [
         5,  2,    2,  5,
        -5,  2,   -2,  5,
         5, -2,    2, -5,
        -5, -2,   -2, -5,
    ]

    //Gradients for 3D. They approximate the directions to the
    //vertices of a rhombicuboctahedron from the center, skewed so
    //that the triangular and square facets can be inscribed inside
    //circles of the same radius.
    let gradients3D: [Int8] = [
        -11,  4,  4,     -4,  11,  4,    -4,  4,  11,
         11,  4,  4,      4,  11,  4,     4,  4,  11,
        -11, -4,  4,     -4, -11,  4,    -4, -4,  11,
         11, -4,  4,      4, -11,  4,     4, -4,  11,
        -11,  4, -4,     -4,  11, -4,    -4,  4, -11,
         11,  4, -4,      4,  11, -4,     4,  4, -11,
        -11, -4, -4,     -4, -11, -4,    -4, -4, -11,
         11, -4, -4,      4, -11, -4,     4, -4, -11,
    ]


    //Gradients for 4D. They approximate the directions to the
    //vertices of a disprismatotesseractihexadecachoron from the center,
    //skewed so that the tetrahedral and cubic faces can be inscribed inside
    //spheres of the same radius.
    let gradients4D: [Int8] = [
         3,  1,  1,  1,      1,  3,  1,  1,      1,  1,  3,  1,      1,  1,  1,  3,
        -3,  1,  1,  1,     -1,  3,  1,  1,     -1,  1,  3,  1,     -1,  1,  1,  3,
         3, -1,  1,  1,      1, -3,  1,  1,      1, -1,  3,  1,      1, -1,  1,  3,
        -3, -1,  1,  1,     -1, -3,  1,  1,     -1, -1,  3,  1,     -1, -1,  1,  3,
         3,  1, -1,  1,      1,  3, -1,  1,      1,  1, -3,  1,      1,  1, -1,  3,
        -3,  1, -1,  1,     -1,  3, -1,  1,     -1,  1, -3,  1,     -1,  1, -1,  3,
         3, -1, -1,  1,      1, -3, -1,  1,      1, -1, -3,  1,      1, -1, -1,  3,
        -3, -1, -1,  1,     -1, -3, -1,  1,     -1, -1, -3,  1,     -1, -1, -1,  3,
         3,  1,  1, -1,      1,  3,  1, -1,      1,  1,  3, -1,      1,  1,  1, -3,
        -3,  1,  1, -1,     -1,  3,  1, -1,     -1,  1,  3, -1,     -1,  1,  1, -3,
         3, -1,  1, -1,      1, -3,  1, -1,      1, -1,  3, -1,      1, -1,  1, -3,
        -3, -1,  1, -1,     -1, -3,  1, -1,     -1, -1,  3, -1,     -1, -1,  1, -3,
         3,  1, -1, -1,      1,  3, -1, -1,      1,  1, -3, -1,      1,  1, -1, -3,
        -3,  1, -1, -1,     -1,  3, -1, -1,     -1,  1, -3, -1,     -1,  1, -1, -3,
         3, -1, -1, -1,      1, -3, -1, -1,      1, -1, -3, -1,      1, -1, -1, -3,
        -3, -1, -1, -1,     -1, -3, -1, -1,     -1, -1, -3, -1,     -1, -1, -1, -3,
    ]

}
