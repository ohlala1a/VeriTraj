(define (domain veritraj-alfworld)
  (:requirements :strips :typing)
  (:types agent place item lamp - object
          receptacle room - place)
  (:constants agent - agent)
  (:predicates
    (at ?a - agent ?p - place)
    (visible ?p - place)
    (in ?o - item ?r - receptacle)
    (in-hand ?o - item)
    (handempty)
    (openable ?r - receptacle)
    (open ?r - receptacle)
    (closed ?r - receptacle)
    (can-heat ?r - receptacle)
    (can-cool ?r - receptacle)
    (can-clean ?r - receptacle)
    (lamp-at ?l - lamp ?r - receptacle)
    (lit ?l - lamp)
    (heated ?o - item)
    (cooled ?o - item)
    (cleaned ?o - item)
    (examined ?o - item))

  (:action goto
    :parameters (?from - place ?to - place)
    :precondition (and (visible ?to))
    :effect (and (not (at agent ?from)) (at agent ?to)))

  (:action open
    :parameters (?r - receptacle)
    :precondition (and (at agent ?r) (openable ?r) (closed ?r))
    :effect (and (not (closed ?r)) (open ?r)))

  (:action close
    :parameters (?r - receptacle)
    :precondition (and (at agent ?r) (openable ?r) (open ?r))
    :effect (and (not (open ?r)) (closed ?r)))

  (:action take
    :parameters (?o - item ?r - receptacle)
    :precondition (and (at agent ?r) (open ?r) (in ?o ?r) (handempty))
    :effect (and (not (in ?o ?r)) (not (handempty)) (in-hand ?o)))

  (:action put
    :parameters (?o - item ?r - receptacle)
    :precondition (and (at agent ?r) (open ?r) (in-hand ?o))
    :effect (and (not (in-hand ?o)) (in ?o ?r) (handempty)))

  (:action heat
    :parameters (?o - item ?r - receptacle)
    :precondition (and (at agent ?r) (can-heat ?r) (in-hand ?o))
    :effect (and (not (cooled ?o)) (heated ?o)))

  (:action cool
    :parameters (?o - item ?r - receptacle)
    :precondition (and (at agent ?r) (can-cool ?r) (in-hand ?o))
    :effect (and (not (heated ?o)) (cooled ?o)))

  (:action clean
    :parameters (?o - item ?r - receptacle)
    :precondition (and (at agent ?r) (can-clean ?r) (in-hand ?o))
    :effect (and (cleaned ?o)))

  (:action toggle
    :parameters (?l - lamp ?r - receptacle)
    :precondition (and (at agent ?r) (lamp-at ?l ?r))
    :effect (and (lit ?l)))

  (:action examine
    :parameters (?o - item ?l - lamp ?r - receptacle)
    :precondition (and (at agent ?r) (lamp-at ?l ?r) (lit ?l) (in-hand ?o))
    :effect (and (examined ?o)))

  (:action look
    :parameters ()
    :precondition (and)
    :effect (and))

  (:action inventory
    :parameters ()
    :precondition (and)
    :effect (and)))
