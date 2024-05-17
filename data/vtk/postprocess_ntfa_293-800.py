# trace generated using paraview version 5.10.0-RC1
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 10

# import the simple module from the paraview
from paraview.simple import *
import glob

# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

directory = r"/home/julius/bwSyncShare/FNO/vtk/"

# create a new 'Legacy VTK Reader'
nTFA293K_fine_temp_293800_ = LegacyVTKReader(
    registrationName="NTFA293K_fine_temp_293-800_*",
    FileNames=glob.glob(directory + "NTFA293K_fine_temp_293-800_*.vtk"),
)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# show data in view
nTFA293K_fine_temp_293800_Display = Show(
    nTFA293K_fine_temp_293800_, renderView1, "UnstructuredGridRepresentation"
)

# get color transfer function/color map for 'DISPLACEMENTS'
dISPLACEMENTSLUT = GetColorTransferFunction("DISPLACEMENTS")

# get opacity transfer function/opacity map for 'DISPLACEMENTS'
dISPLACEMENTSPWF = GetOpacityTransferFunction("DISPLACEMENTS")

# trace defaults for the display properties.
nTFA293K_fine_temp_293800_Display.Representation = "Surface"
nTFA293K_fine_temp_293800_Display.ColorArrayName = ["POINTS", "DISPLACEMENTS"]
nTFA293K_fine_temp_293800_Display.LookupTable = dISPLACEMENTSLUT
nTFA293K_fine_temp_293800_Display.SelectTCoordArray = "None"
nTFA293K_fine_temp_293800_Display.SelectNormalArray = "None"
nTFA293K_fine_temp_293800_Display.SelectTangentArray = "None"
nTFA293K_fine_temp_293800_Display.OSPRayScaleArray = "DISPLACEMENTS"
nTFA293K_fine_temp_293800_Display.OSPRayScaleFunction = "PiecewiseFunction"
nTFA293K_fine_temp_293800_Display.SelectOrientationVectors = "DISPLACEMENTS"
nTFA293K_fine_temp_293800_Display.ScaleFactor = 11.0
nTFA293K_fine_temp_293800_Display.SelectScaleArray = "DISPLACEMENTS"
nTFA293K_fine_temp_293800_Display.GlyphType = "Arrow"
nTFA293K_fine_temp_293800_Display.GlyphTableIndexArray = "DISPLACEMENTS"
nTFA293K_fine_temp_293800_Display.GaussianRadius = 0.55
nTFA293K_fine_temp_293800_Display.SetScaleArray = ["POINTS", "DISPLACEMENTS"]
nTFA293K_fine_temp_293800_Display.ScaleTransferFunction = "PiecewiseFunction"
nTFA293K_fine_temp_293800_Display.OpacityArray = ["POINTS", "DISPLACEMENTS"]
nTFA293K_fine_temp_293800_Display.OpacityTransferFunction = "PiecewiseFunction"
nTFA293K_fine_temp_293800_Display.DataAxesGrid = "GridAxesRepresentation"
nTFA293K_fine_temp_293800_Display.PolarAxes = "PolarAxesRepresentation"
nTFA293K_fine_temp_293800_Display.ScalarOpacityFunction = dISPLACEMENTSPWF
nTFA293K_fine_temp_293800_Display.ScalarOpacityUnitDistance = 2.6566498575704793
nTFA293K_fine_temp_293800_Display.OpacityArrayName = ["POINTS", "DISPLACEMENTS"]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
nTFA293K_fine_temp_293800_Display.ScaleTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    1.1757813367477812e-38,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
nTFA293K_fine_temp_293800_Display.OpacityTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    1.1757813367477812e-38,
    1.0,
    0.5,
    0.0,
]

# reset view to fit data
renderView1.ResetCamera(False)

# show color bar/color legend
nTFA293K_fine_temp_293800_Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

animationScene1.GoToLast()

# set scalar coloring
ColorBy(nTFA293K_fine_temp_293800_Display, ("CELLS", "QBAR"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(dISPLACEMENTSLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
nTFA293K_fine_temp_293800_Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
nTFA293K_fine_temp_293800_Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'QBAR'
qBARLUT = GetColorTransferFunction("QBAR")

# get opacity transfer function/opacity map for 'QBAR'
qBARPWF = GetOpacityTransferFunction("QBAR")

# Rescale transfer function
qBARLUT.RescaleTransferFunction(0.01, 0.08)

# Rescale transfer function
qBARPWF.RescaleTransferFunction(0.01, 0.08)

# Rescale transfer function
qBARLUT.RescaleTransferFunction(0.01, 0.06)

# Rescale transfer function
qBARPWF.RescaleTransferFunction(0.01, 0.06)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
qBARLUT.ApplyPreset("Rainbow Uniform", True)

# get color legend/bar for qBARLUT in view renderView1
qBARLUTColorBar = GetScalarBar(qBARLUT, renderView1)

# change scalar bar placement
qBARLUTColorBar.WindowLocation = "Any Location"
qBARLUTColorBar.Position = [0.8985637342908438, 0.32594235033259417]
qBARLUTColorBar.ScalarBarLength = 0.33000000000000035

# Properties modified on qBARLUTColorBar
qBARLUTColorBar.Title = "Q"
qBARLUTColorBar.TitleFontFamily = "Times"
qBARLUTColorBar.LabelFontFamily = "Times"

# hide color bar/color legend
nTFA293K_fine_temp_293800_Display.SetScalarBarVisibility(renderView1, False)

# show color bar/color legend
nTFA293K_fine_temp_293800_Display.SetScalarBarVisibility(renderView1, True)

# Properties modified on qBARLUTColorBar
qBARLUTColorBar.ScalarBarThickness = 48

# change scalar bar placement
qBARLUTColorBar.Position = [0.8138597869224228, 0.3181818181818181]
qBARLUTColorBar.ScalarBarLength = 0.3300000000000005

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 0

# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# get layout
layout1 = GetLayout()

# --------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1216, 902)

# -----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [0.0, 0.0, 269.2767700257741]
renderView1.CameraFocalPoint = [0.0, 0.0, 2.5]
renderView1.CameraViewUp = [-1.0, 2.220446049250313e-16, 0.0]
renderView1.CameraParallelScale = 57.063561052566634

# --------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
