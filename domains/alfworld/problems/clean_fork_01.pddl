(define (problem clean-fork-01)
  (:domain veritraj-alfworld)
  (:objects
    room_1 - room
    cabinet_1 countertop_1 countertop_2 diningtable_1 drawer_1 fridge_1 sinkbasin_1 - receptacle
    bread_1 cup_1 fork_1 saltshaker_1 spoon_1 - item)
  (:init
    (at agent room_1) (handempty)
    (visible cabinet_1) (visible countertop_1) (visible countertop_2) (visible diningtable_1)
    (visible drawer_1) (visible fridge_1) (visible sinkbasin_1)
    (openable cabinet_1) (openable drawer_1) (openable fridge_1)
    (closed cabinet_1) (closed drawer_1) (closed fridge_1)
    (open countertop_1) (open countertop_2) (open diningtable_1) (open sinkbasin_1)
    (can-clean sinkbasin_1) (can-cool fridge_1)
    (in fork_1 diningtable_1) (in saltshaker_1 diningtable_1) (in spoon_1 drawer_1)
    (in bread_1 countertop_1) (in cup_1 countertop_2))
  (:goal (and (cleaned fork_1) (in fork_1 drawer_1))))
