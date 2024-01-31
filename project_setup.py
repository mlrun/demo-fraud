# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import mlrun

def setup(
        project: mlrun.projects.MlrunProject
) -> mlrun.projects.MlrunProject:
    """
    Creating the project for this demo. This function is expected to be called automatically when
    calling the function `mlrun.get_or_create_project`.

    :returns: a fully prepared project for this demo.
    """
    # Set the project git source:
    source = project.get_param(key="source")
    if source:
        print(f"Project Source: {source}")
        project.set_source(source=source, pull_at_runtime=True)

    if project.get_param("pre_load_data"):
        print("pre_load_data")


    # Set the training workflow:
    project.set_workflow("main", "src/train_workflow.py")

    # Save and return the project:
    project.save()
    return project
