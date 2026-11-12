(define (problem pick-two-cellphone-01)
  (:domain veritraj-alfworld)
  (:objects
    room_1 - room
    armchair_1 coffeetable_1 drawer_1 garbagecan_1 shelf_1 shelf_2 sofa_1 - receptacle
    cellphone_1 cellphone_2 creditcard_1 pillow_1 remotecontrol_1 - item)
  (:init
    (at agent room_1) (handempty)
    (visible armchair_1) (visible coffeetable_1) (visible drawer_1) (visible garbagecan_1)
    (visible shelf_1) (visible shelf_2) (visible sofa_1)
    (openable drawer_1) (closed drawer_1)
    (open armchair_1) (open coffeetable_1) (open garbagecan_1) (open shelf_1) (open shelf_2) (open sofa_1)
    (in cellphone_1 sofa_1) (in remotecontrol_1 sofa_1) (in pillow_1 armchair_1)
    (in cellphone_2 coffeetable_1) (in creditcard_1 drawer_1))
  (:goal (and (in cellphone_1 shelf_1) (in cellphone_2 shelf_1))))
