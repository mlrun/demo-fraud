import mlrun

def setup(
        project: mlrun.projects.MlrunProject
):
    """
    Creating the project for this demo.
    :returns: a fully prepared project for this demo.
    """
    print(project.get_param("source"))

    # Set the project git source:

    project.set_source(project.get_param("source"), pull_at_runtime=True)

    if project.get_param("pre_load_data"):
        continue


    # Set the training workflow:
    project.set_workflow("main", "src/new_train_workflow.py")

    # Save and return the project:
    project.save()
    return project
