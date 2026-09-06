// archmage ebc204640c64f407cee9f6521b5b2e1385d0dbd6; Rust 1.98.0; aarch64-apple-darwin
// Extracted from idiomatic_patterns_all release assembly, lines 402-425.
	stp	q0, q1, [x0, #32]
	movi.2d	v0, #0000000000000000
	ldp	q1, q2, [x19]
	ldp	q3, q4, [x0]
	fmla.4s	v0, v3, v1
	ldp	q1, q3, [x19, #32]
	ldp	q5, q6, [x0, #32]
	fmla.4s	v0, v5, v1
	mov	w8, #1107820544
	str	w8, [x0, #64]
	mov	w8, #8192
	movk	w8, #17759, lsl #16
	str	w8, [sp, #32]
	movi.2d	v1, #0000000000000000
	fmla.4s	v1, v4, v2
	fmla.4s	v1, v6, v3
	fadd.4s	v0, v0, v1
	faddp.4s	v0, v0, v0
	faddp.2s	s0, v0
	mov	w8, #32768
	movk	w8, #17424, lsl #16
	fmov	s1, w8
	fadd	s0, s0, s1
	str	s0, [sp, #36]
