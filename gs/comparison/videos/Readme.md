# Comparison
## SCD vs XPBD (Stiffness Ratio) 
N = 5, First Compliance = 1.0e-4
Left: SCD
Right: XPBD
`scdvsXPBD_stiffness.mp4`

## SCD vs XPBD (Mass Ratio)


## SCD vs XPBD (Jacobi Solver)
N = 5
Left: SCD
Right: XPBD
`scdvsXPBD_Jacobi.mp4`

## Geometric Stiffness (Enabled)
N = 5
Left: Enable
Right: Disable
`scd_GSEnabled.mp4`

## K vs Diag(K) Uniform Mass
N = 5
Left : Full K
Right: Diag K
`scd_DiagKComparison.mp4`

## K vs Diag(K) with Big Mass
N = 5, Last Mass = 100.0
Left : Full K
Right: Diag K
`scd_DiagKBigMassComparison.mp4`

## Zeros vs NonZeros (Upper of RHS)
N = 5
Left: Upper right hand size is zeros
Right: Upper right hand size is not zerors
`scd_UpperRHSIFZeros.mp4`

## CG vs LLT
N = 5, Last Mass = 100.0, CGIte = 1, Remove Indefinite Part of stiffness matrix K
`scd_CGvsLLTBigMassComparison.mp4`


## CG Cube Mesh
