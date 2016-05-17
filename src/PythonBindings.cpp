/*
 * PythonBindings.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: peter
 */

#include "PythonBindings.h"


void import_array_wrapper() {
 _import_array();
 std::cout << "IMPORT..." << std::endl;
}
