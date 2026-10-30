package dev.sightseen.request;

public enum RequestStatus {
    /** The student has the landlord link; nothing filmed yet. */
    WAITING_FOR_VIDEO,
    /** A video is saved and about to go to the 3D helper. */
    VIDEO_RECEIVED,
    /** MapAnything is rebuilding the room. */
    RECONSTRUCTING,
    /** The 3D room is ready for the student. */
    READY,
    /** Something stopped it; the request's failure message says what. */
    FAILED
}
