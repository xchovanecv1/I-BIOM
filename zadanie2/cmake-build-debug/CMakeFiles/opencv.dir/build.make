# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "E:\Programy\JetBrains\CLion 2018.3.4\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "E:\Programy\JetBrains\CLion 2018.3.4\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\Skola\BIOM\zadanie2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\Skola\BIOM\zadanie2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\opencv.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\opencv.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\opencv.dir\flags.make

CMakeFiles\opencv.dir\main.cpp.obj: CMakeFiles\opencv.dir\flags.make
CMakeFiles\opencv.dir\main.cpp.obj: ..\main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=E:\Skola\BIOM\zadanie2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv.dir/main.cpp.obj"
	C:\PROGRA~2\MICROS~1\2017\ENTERP~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\opencv.dir\main.cpp.obj /FdCMakeFiles\opencv.dir\ /FS -c E:\Skola\BIOM\zadanie2\main.cpp
<<

CMakeFiles\opencv.dir\main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/main.cpp.i"
	C:\PROGRA~2\MICROS~1\2017\ENTERP~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\cl.exe > CMakeFiles\opencv.dir\main.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\Skola\BIOM\zadanie2\main.cpp
<<

CMakeFiles\opencv.dir\main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/main.cpp.s"
	C:\PROGRA~2\MICROS~1\2017\ENTERP~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\opencv.dir\main.cpp.s /c E:\Skola\BIOM\zadanie2\main.cpp
<<

# Object files for target opencv
opencv_OBJECTS = \
"CMakeFiles\opencv.dir\main.cpp.obj"

# External object files for target opencv
opencv_EXTERNAL_OBJECTS =

opencv.exe: CMakeFiles\opencv.dir\main.cpp.obj
opencv.exe: CMakeFiles\opencv.dir\build.make
opencv.exe: E:\opencv\build\lib\Debug\opencv_gapi401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_stitching401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_aruco401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_bgsegm401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_bioinspired401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_ccalib401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_dnn_objdetect401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_dpm401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_face401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_fuzzy401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_hfs401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_img_hash401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_line_descriptor401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_reg401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_rgbd401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_stereo401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_structured_light401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_superres401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_surface_matching401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_tracking401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_videostab401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_xfeatures2d401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_xobjdetect401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_xphoto401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_shape401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_phase_unwrapping401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_optflow401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_ximgproc401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_datasets401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_plot401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_text401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_dnn401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_ml401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_video401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_objdetect401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_calib3d401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_features2d401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_flann401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_highgui401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_videoio401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_imgcodecs401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_photo401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_imgproc401d.lib
opencv.exe: E:\opencv\build\lib\Debug\opencv_core401d.lib
opencv.exe: CMakeFiles\opencv.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=E:\Skola\BIOM\zadanie2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencv.exe"
	"E:\Programy\JetBrains\CLion 2018.3.4\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\opencv.dir --manifests  -- C:\PROGRA~2\MICROS~1\2017\ENTERP~1\VC\Tools\MSVC\1412~1.258\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\opencv.dir\objects1.rsp @<<
 /out:opencv.exe /implib:opencv.lib /pdb:E:\Skola\BIOM\zadanie2\cmake-build-debug\opencv.pdb /version:0.0  /machine:x64 /debug /INCREMENTAL /subsystem:console E:\opencv\build\lib\Debug\opencv_gapi401d.lib E:\opencv\build\lib\Debug\opencv_stitching401d.lib E:\opencv\build\lib\Debug\opencv_aruco401d.lib E:\opencv\build\lib\Debug\opencv_bgsegm401d.lib E:\opencv\build\lib\Debug\opencv_bioinspired401d.lib E:\opencv\build\lib\Debug\opencv_ccalib401d.lib E:\opencv\build\lib\Debug\opencv_dnn_objdetect401d.lib E:\opencv\build\lib\Debug\opencv_dpm401d.lib E:\opencv\build\lib\Debug\opencv_face401d.lib E:\opencv\build\lib\Debug\opencv_fuzzy401d.lib E:\opencv\build\lib\Debug\opencv_hfs401d.lib E:\opencv\build\lib\Debug\opencv_img_hash401d.lib E:\opencv\build\lib\Debug\opencv_line_descriptor401d.lib E:\opencv\build\lib\Debug\opencv_reg401d.lib E:\opencv\build\lib\Debug\opencv_rgbd401d.lib E:\opencv\build\lib\Debug\opencv_stereo401d.lib E:\opencv\build\lib\Debug\opencv_structured_light401d.lib E:\opencv\build\lib\Debug\opencv_superres401d.lib E:\opencv\build\lib\Debug\opencv_surface_matching401d.lib E:\opencv\build\lib\Debug\opencv_tracking401d.lib E:\opencv\build\lib\Debug\opencv_videostab401d.lib E:\opencv\build\lib\Debug\opencv_xfeatures2d401d.lib E:\opencv\build\lib\Debug\opencv_xobjdetect401d.lib E:\opencv\build\lib\Debug\opencv_xphoto401d.lib E:\opencv\build\lib\Debug\opencv_shape401d.lib E:\opencv\build\lib\Debug\opencv_phase_unwrapping401d.lib E:\opencv\build\lib\Debug\opencv_optflow401d.lib E:\opencv\build\lib\Debug\opencv_ximgproc401d.lib E:\opencv\build\lib\Debug\opencv_datasets401d.lib E:\opencv\build\lib\Debug\opencv_plot401d.lib E:\opencv\build\lib\Debug\opencv_text401d.lib E:\opencv\build\lib\Debug\opencv_dnn401d.lib E:\opencv\build\lib\Debug\opencv_ml401d.lib E:\opencv\build\lib\Debug\opencv_video401d.lib E:\opencv\build\lib\Debug\opencv_objdetect401d.lib E:\opencv\build\lib\Debug\opencv_calib3d401d.lib E:\opencv\build\lib\Debug\opencv_features2d401d.lib E:\opencv\build\lib\Debug\opencv_flann401d.lib E:\opencv\build\lib\Debug\opencv_highgui401d.lib E:\opencv\build\lib\Debug\opencv_videoio401d.lib E:\opencv\build\lib\Debug\opencv_imgcodecs401d.lib E:\opencv\build\lib\Debug\opencv_photo401d.lib E:\opencv\build\lib\Debug\opencv_imgproc401d.lib E:\opencv\build\lib\Debug\opencv_core401d.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\opencv.dir\build: opencv.exe

.PHONY : CMakeFiles\opencv.dir\build

CMakeFiles\opencv.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\opencv.dir\cmake_clean.cmake
.PHONY : CMakeFiles\opencv.dir\clean

CMakeFiles\opencv.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" E:\Skola\BIOM\zadanie2 E:\Skola\BIOM\zadanie2 E:\Skola\BIOM\zadanie2\cmake-build-debug E:\Skola\BIOM\zadanie2\cmake-build-debug E:\Skola\BIOM\zadanie2\cmake-build-debug\CMakeFiles\opencv.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\opencv.dir\depend

