# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/snustorm/projects/micrograd_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/snustorm/projects/micrograd_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/micrograd_cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/micrograd_cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/micrograd_cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/micrograd_cpp.dir/flags.make

CMakeFiles/micrograd_cpp.dir/test.cpp.o: CMakeFiles/micrograd_cpp.dir/flags.make
CMakeFiles/micrograd_cpp.dir/test.cpp.o: /home/snustorm/projects/micrograd_cpp/test.cpp
CMakeFiles/micrograd_cpp.dir/test.cpp.o: CMakeFiles/micrograd_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/snustorm/projects/micrograd_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/micrograd_cpp.dir/test.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/micrograd_cpp.dir/test.cpp.o -MF CMakeFiles/micrograd_cpp.dir/test.cpp.o.d -o CMakeFiles/micrograd_cpp.dir/test.cpp.o -c /home/snustorm/projects/micrograd_cpp/test.cpp

CMakeFiles/micrograd_cpp.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/micrograd_cpp.dir/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/snustorm/projects/micrograd_cpp/test.cpp > CMakeFiles/micrograd_cpp.dir/test.cpp.i

CMakeFiles/micrograd_cpp.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/micrograd_cpp.dir/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/snustorm/projects/micrograd_cpp/test.cpp -o CMakeFiles/micrograd_cpp.dir/test.cpp.s

CMakeFiles/micrograd_cpp.dir/value.cpp.o: CMakeFiles/micrograd_cpp.dir/flags.make
CMakeFiles/micrograd_cpp.dir/value.cpp.o: /home/snustorm/projects/micrograd_cpp/value.cpp
CMakeFiles/micrograd_cpp.dir/value.cpp.o: CMakeFiles/micrograd_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/snustorm/projects/micrograd_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/micrograd_cpp.dir/value.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/micrograd_cpp.dir/value.cpp.o -MF CMakeFiles/micrograd_cpp.dir/value.cpp.o.d -o CMakeFiles/micrograd_cpp.dir/value.cpp.o -c /home/snustorm/projects/micrograd_cpp/value.cpp

CMakeFiles/micrograd_cpp.dir/value.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/micrograd_cpp.dir/value.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/snustorm/projects/micrograd_cpp/value.cpp > CMakeFiles/micrograd_cpp.dir/value.cpp.i

CMakeFiles/micrograd_cpp.dir/value.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/micrograd_cpp.dir/value.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/snustorm/projects/micrograd_cpp/value.cpp -o CMakeFiles/micrograd_cpp.dir/value.cpp.s

CMakeFiles/micrograd_cpp.dir/ops.cpp.o: CMakeFiles/micrograd_cpp.dir/flags.make
CMakeFiles/micrograd_cpp.dir/ops.cpp.o: /home/snustorm/projects/micrograd_cpp/ops.cpp
CMakeFiles/micrograd_cpp.dir/ops.cpp.o: CMakeFiles/micrograd_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/snustorm/projects/micrograd_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/micrograd_cpp.dir/ops.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/micrograd_cpp.dir/ops.cpp.o -MF CMakeFiles/micrograd_cpp.dir/ops.cpp.o.d -o CMakeFiles/micrograd_cpp.dir/ops.cpp.o -c /home/snustorm/projects/micrograd_cpp/ops.cpp

CMakeFiles/micrograd_cpp.dir/ops.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/micrograd_cpp.dir/ops.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/snustorm/projects/micrograd_cpp/ops.cpp > CMakeFiles/micrograd_cpp.dir/ops.cpp.i

CMakeFiles/micrograd_cpp.dir/ops.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/micrograd_cpp.dir/ops.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/snustorm/projects/micrograd_cpp/ops.cpp -o CMakeFiles/micrograd_cpp.dir/ops.cpp.s

# Object files for target micrograd_cpp
micrograd_cpp_OBJECTS = \
"CMakeFiles/micrograd_cpp.dir/test.cpp.o" \
"CMakeFiles/micrograd_cpp.dir/value.cpp.o" \
"CMakeFiles/micrograd_cpp.dir/ops.cpp.o"

# External object files for target micrograd_cpp
micrograd_cpp_EXTERNAL_OBJECTS =

micrograd_cpp: CMakeFiles/micrograd_cpp.dir/test.cpp.o
micrograd_cpp: CMakeFiles/micrograd_cpp.dir/value.cpp.o
micrograd_cpp: CMakeFiles/micrograd_cpp.dir/ops.cpp.o
micrograd_cpp: CMakeFiles/micrograd_cpp.dir/build.make
micrograd_cpp: CMakeFiles/micrograd_cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/snustorm/projects/micrograd_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable micrograd_cpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/micrograd_cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/micrograd_cpp.dir/build: micrograd_cpp
.PHONY : CMakeFiles/micrograd_cpp.dir/build

CMakeFiles/micrograd_cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/micrograd_cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/micrograd_cpp.dir/clean

CMakeFiles/micrograd_cpp.dir/depend:
	cd /home/snustorm/projects/micrograd_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/snustorm/projects/micrograd_cpp /home/snustorm/projects/micrograd_cpp /home/snustorm/projects/micrograd_cpp/build /home/snustorm/projects/micrograd_cpp/build /home/snustorm/projects/micrograd_cpp/build/CMakeFiles/micrograd_cpp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/micrograd_cpp.dir/depend

