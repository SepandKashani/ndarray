// ############################################################################
// test.cpp
// ========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_CPP
#define TEST_CPP

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif // TEST_CPP
