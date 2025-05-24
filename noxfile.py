# Copyright (c) 2025 David Boetius
# Licensed under the MIT license
import nox

@nox.session(python=["3.10", "3.12"])
@nox.parametrize("versions", ["torch==1.12.1 numpy>=1.25,<1.26", "torch==2.7 numpy>=2.0"])
def tests(session, versions):
    session.install(".", *versions.split(" "))
    session.install("pytest")
    session.run("pytest")

