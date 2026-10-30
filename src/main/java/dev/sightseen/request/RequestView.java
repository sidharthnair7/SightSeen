package dev.sightseen.request;

import java.time.Instant;

/** What the student's page sees. The landlord gets FilmController.FilmView instead, which holds much less. */
public record RequestView(String id, String label, RequestStatus status, String stage, String landlordPath,
                          String roomPath, Instant createdAt, VideoSource videoSource, String model, Room room,
                          String failureMessage) {

    /** The last finished 3D room, if there is one. */
    public record Room(int frames, double seconds, Double peakVramGb, Double scaleFactor, long points,
                       double sizeX, double sizeY, double sizeZ, String plyUrl) {
    }

    static RequestView of(RoomRequest r) {
        Room room = r.hasRoom()
                ? new Room(r.getFrames(), r.getSeconds(), r.getPeakVramGb(), r.getScaleFactor(), r.getPoints(),
                r.getSizeX(), r.getSizeY(), r.getSizeZ(),
                "/api/requests/" + r.getId() + "/room.ply?v=" + r.getReadyAt().toEpochMilli())
                : null;
        return new RequestView(r.getId(), r.getLabel(), r.getStatus(), r.getStage(),
                "/film/" + r.getLandlordToken(), "/room/" + r.getId(), r.getCreatedAt(), r.getVideoSource(),
                r.getModel(), room, r.getFailureMessage());
    }
}
