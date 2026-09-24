
set(OPENQCD_MODULE_DIRS
    archive
    block
    dfl
    dft
    dirac
    flags
    forces
    lattice
    linalg
    linsolv
    little
    mdflds
    msfcts
    random
    ratfcts
    sap
    sflds
    su3fcts
    sw_term
    tcharge
    uflds
    update
    utils
    vflds
    wflow
)

set(OPENQCD_SOURCES "")
foreach(dir ${OPENQCD_MODULE_DIRS})
    file(GLOB dir_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.c)
    list(APPEND OPENQCD_SOURCES ${dir_sources})
endforeach()

add_library(openqcd STATIC ${OPENQCD_SOURCES})

target_include_directories(openqcd PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# enable FMA3/PM, set as flag
if(OPENQCD_ENABLE_FMA3)
    target_compile_definitions(openqcd PUBLIC FMA3)
endif()

if(OPENQCD_ENABLE_PM)
    target_compile_definitions(openqcd PUBLIC PM)
endif()

target_compile_options(openqcd PRIVATE
    -pedantic -fstrict-aliasing -Wall -Wno-long-long -Wstrict-prototypes -O
)

target_link_libraries(openqcd PUBLIC MPI::MPI_C OpenMP::OpenMP_C m)

# Helper functions used by the devel check programs
# (the DEVFCTS module list in the devel/*/Makefile files)
file(GLOB DEVFCTS_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/devfcts/*.c)

add_library(openqcd_devfcts STATIC ${DEVFCTS_SOURCES})

target_compile_options(openqcd_devfcts PRIVATE
    -pedantic -fstrict-aliasing -Wall -Wno-long-long -Wstrict-prototypes -O
)

target_link_libraries(openqcd_devfcts PUBLIC openqcd)
