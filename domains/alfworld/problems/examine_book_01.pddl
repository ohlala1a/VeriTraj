; family: examine
; task: look at book under the desklamp.
(define (problem examine-book-01)
  (:domain veritraj-alfworld)
  (:objects
    room_1 - room
    drawer_1 drawer_2 dresser_1 garbagecan_1 shelf_1 sidetable_1 - receptacle
    desklamp_1 - lamp
    book_1 cd_1 keychain_1 pen_1 - item)
  (:init
    (at agent room_1) (handempty)
    (visible drawer_1) (visible drawer_2) (visible dresser_1) (visible garbagecan_1)
    (visible shelf_1) (visible sidetable_1)
    (openable drawer_1) (openable drawer_2) (closed drawer_1) (closed drawer_2)
    (open dresser_1) (open garbagecan_1) (open shelf_1) (open sidetable_1)
    (lamp-at desklamp_1 sidetable_1)
    (in book_1 drawer_2) (in cd_1 shelf_1) (in pen_1 dresser_1) (in keychain_1 sidetable_1))
  (:goal (and (examined book_1))))
