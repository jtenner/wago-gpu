package wagogpu

// Integer-only conversion preserves zero signs and canonicalizes scalar NaNs.
func halfToFloat(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	e := uint32(h>>10) & 31
	m := uint32(h & 1023)
	if e == 31 {
		if m != 0 {
			return 0x7fc00000
		}
		return sign | 0x7f800000
	}
	if e == 0 {
		if m == 0 {
			return sign
		}
		e = 113
		for m&1024 == 0 {
			m <<= 1
			e--
		}
		return sign | e<<23 | (m&1023)<<13
	}
	return sign | (e+112)<<23 | m<<13
}
func floatToHalf(bits uint32) uint16 {
	sign := uint16(bits>>16) & 0x8000
	e := (bits >> 23) & 255
	m := bits & 0x7fffff
	if e == 255 {
		if m != 0 {
			return 0x7e00
		}
		return sign | 0x7c00
	}
	if e > 142 {
		return sign | 0x7c00
	}
	if e < 102 {
		return sign
	}
	if e < 113 {
		m |= 0x800000
		shift := uint32(126 - e)
		v := m >> shift
		rem := m & ((1 << shift) - 1)
		mid := uint32(1) << (shift - 1)
		if rem > mid || rem == mid && v&1 != 0 {
			v++
		}
		return sign | uint16(v)
	}
	v := ((e - 112) << 10) | (m >> 13)
	rem := m & 8191
	if rem > 4096 || rem == 4096 && v&1 != 0 {
		v++
	}
	return sign | uint16(v)
}

const halfWGSL = `
fn half_to_f32(h: u32) -> u32 {
 let sign = (h & 32768u) << 16u;
 var e = (h >> 10u) & 31u;
 var m = h & 1023u;
 if (e == 31u) { if (m != 0u) { return 0x7fc00000u; } return sign | 0x7f800000u; }
 if (e == 0u) {
  if (m == 0u) { return sign; }
  e = 113u;
  loop { if ((m & 1024u) != 0u) { break; } m = m << 1u; e = e - 1u; }
  return sign | (e << 23u) | ((m & 1023u) << 13u);
 }
 return sign | ((e + 112u) << 23u) | (m << 13u);
}
fn f32_to_half(bits: u32) -> u32 {
 let sign = (bits >> 16u) & 32768u;
 let e = (bits >> 23u) & 255u;
 var m = bits & 0x7fffffu;
 if (e == 255u) { if (m != 0u) { return 0x7e00u; } return sign | 0x7c00u; }
 if (e > 142u) { return sign | 0x7c00u; }
 if (e < 102u) { return sign; }
 if (e < 113u) {
  m = m | 0x800000u;
  let shift = 126u - e;
  var v = m >> shift;
  let rem = m & ((1u << shift) - 1u);
  let mid = 1u << (shift - 1u);
  if (rem > mid || (rem == mid && (v & 1u) != 0u)) { v = v + 1u; }
  return sign | v;
 }
 var v = ((e - 112u) << 10u) | (m >> 13u);
 let rem = m & 8191u;
 if (rem > 4096u || (rem == 4096u && (v & 1u) != 0u)) { v = v + 1u; }
 return sign | v;
}
`
