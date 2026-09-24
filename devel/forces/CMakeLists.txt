

function(openqcd_add_exec name)
    add_executable(${name} ${name}.c)
    target_link_libraries(${name} PRIVATE openqcd openqcd_devfcts)
    target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/devel)
endfunction()

openqcd_add_exec(check1)
openqcd_add_exec(check2)
openqcd_add_exec(check3)
openqcd_add_exec(check4)
openqcd_add_exec(check5)
openqcd_add_exec(check6)
openqcd_add_exec(check7)
openqcd_add_exec(check8)
openqcd_add_exec(check9)
openqcd_add_exec(check10)
openqcd_add_exec(check11)
openqcd_add_exec(check12)
openqcd_add_exec(time1)

# Copy any .in files to build folder
file(GLOB IN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.in")
file(COPY ${IN_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
