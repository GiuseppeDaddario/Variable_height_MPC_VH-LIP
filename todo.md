## Todo list
- [x] Update from LIP to VH-LIP 
    - [ ] Test it
- [ ] Through multiple iterations, find the best a_max
- [ ] Implement MPC
    - [ ] Implement QP-z
    - [ ] Implement QP-xy
    - [ ] Integrate them together
    - [ ] Test the whole thing
- [ ] Test walking 
    - [ ] On flat ground
    - [ ] On different surfaces
    - [ ] On flat ground with different heights!!
- [ ] Test running
    - [ ] On flat ground
    - [ ] On different surfaces

### Steps QP-z
[X] Decisional variables
[X] Parameters
[X] Dynamics
[X] Cost function
[X] Constraints
    [X] No slipping
    [X] 

### Steps QP-xy
[X] Decisional variables
[X] Parameters
[X] Dynamics
[X] Cost function
[ ] Constraints
    [X] ZMP constraint
    [ ] Ground patch constraint
    [ ] Kinematic constraint
    [ ] Swing foot constraint
    [ ] Stability constraint
[ ] QP-xy function
[ ] Pipeline in solve()