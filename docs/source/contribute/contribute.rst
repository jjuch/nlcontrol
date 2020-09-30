.. _contribution:

==========================
Contribute - Git workflow
==========================


.. only:: html

    .. contents::
       :depth: 3
       :backlinks: none

This is just a practical guide that can help you making contributions to the nlcontrol toolbox. It is very basic, so don't expect too much.

Commit message
---------------

Refer to a component name, give a short description, and add a reference to the issue, if relevant (with 'fix #<number>' it means it is fixed)
::

    COMPONENT_NAME: fix *some_text* (fix #1234)

    More details here...


Initiate your work repository
------------------------------

Fork the jjuch/nlcontrol from github UI, and then
::

    git clone https://github.com/jjuch/nlcontrol.git
    cd nlcontrol
    git remote add my_user_name https://github.com/my_user_name/nlcontrol.git


Update your local master against upstream master
----------------------------------------------------------

In command line do the following
::

    git checkout master
    git fetch origin
    # Be careful: this will remove all local changes you might have done now
    git reset --hard origin/master

Working with a feature branch
------------------------------

In command line do the following
::

    git checkout master
    (potentially update your local master against upstream, as described above)
    git checkout -b my_new_feature_branch

    # do something. For instance:
    git add my_new_file
    git add my_modified_message
    git rm old_file
    git commit -a 

    # you may need to resynchronize against master if you need some bugfix
    # or new capability that has been added since you created your branch
    git fetch origin
    git rebase origin/master

    # At end of your work, make sure history is reasonable by folding non
    # significant commits into a consistent set
    git rebase -i master (use 'fixup' for example to merge several commits together,
    and 'reword' to modify commit messages)

    # or alternatively, in case there is a big number of commits and marking
    # all them as 'fixup' is tedious
    git fetch origin
    git rebase origin/master
    git reset --soft origin/master
    git commit -a -m "Put here the synthetic commit message"

    # push your branch
    git push my_user_name my_new_feature_branch
    From GitHub UI, issue a pull request

If the pull request discussion checks 'requires changes', commit locally and push. To get a clean history, you may need to ``git rebase -i master``, in which case you will have to force-push your branch with ``git push -f my_user_name my_new_feature_branch``.


Things you should NOT do
-------------------------
(For anyone with push rights to https://github.com/jjuch/nlcontrol,) Never modify a commit or the history of anything that has been committed to https://github.com/jjuch/nlcontrol


**Disclaimer:** Thank you GDAL repo for the inspiration.