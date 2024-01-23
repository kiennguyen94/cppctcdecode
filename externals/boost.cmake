
include( ExternalProject )
message( "External project - Boost" )

set( Boost_Bootstrap_Command )
if( UNIX )
  set( Boost_Bootstrap_Command boost_1_83_0/bootstrap.sh && cp b2 ./boost_1_83_0/)
  set( Boost_b2_Command cd ./boost_1_83_0/ && ./b2 )
else()
  if( WIN32 )
    set( Boost_Bootstrap_Command bootstrap.bat )
    set( Boost_b2_Command b2.exe )
  endif()
endif()

ExternalProject_Add(boost
  URL "https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.gz"
  BUILD_IN_SOURCE 1
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ${Boost_Bootstrap_Command}
  BUILD_COMMAND  ${Boost_b2_Command} install
    --without-python
    --without-mpi
    --disable-icu
    --prefix=${CMAKE_BINARY_DIR}/INSTALL
    --threading=single,multi
    --link=shared
    --variant=release
    -j8
  INSTALL_COMMAND ""
  INSTALL_DIR ${CMAKE_BINARY_DIR}/INSTALL
)


if( NOT WIN32 )
  set(Boost_LIBRARY_DIR ${CMAKE_BINARY_DIR}/INSTALL/lib/boost/ )
  set(Boost_INCLUDE_DIR ${CMAKE_BINARY_DIR}/INSTALL/include/ )
else()
  set(Boost_LIBRARY_DIR ${CMAKE_BINARY_DIR}/INSTALL/lib/ )
  set(Boost_INCLUDE_DIR ${CMAKE_BINARY_DIR}/INSTALL/include/boost-1_49/ )
endif()