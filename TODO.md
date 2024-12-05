# Overall
- Look into something more efficent for storing a tetrahedra. We are wasting floats
# Vertex shader
- Vertex shader spits out circumcenters
- Circumecenters get plugged into TCNN
- Adjust data going into vertex shader (proj_mat and wvt - both can't be necessary)
- Adjust rectangle calculations - going to need to project points to screen
# Tile shader
- Nothing
# Alpha Blend Shader
- Here is the bulk of the work.
- Remove skipping of "splats"
- Adjust all inputs and outputs
- load\_splat\_alphablend -> load\_tetrahedra
- splat\_tiled just needs to adjust inputs and outputs
- bwd\_alpha\_blend just needs to adjust inputs, outputs, and function names. Need to adjust all the different backprop function and storage names too
- evaluate\_splat is going to be the difficult function to change
- everything else is pretty straightforward.

# Plan
3. Build evaluate\_tetra
5. Refactor code to use these new functions and data using Claude
6. Plug in diagram to render using Delaunay from scipy
