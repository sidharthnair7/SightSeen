package dev.sightseen.request;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;

import java.security.SecureRandom;
import java.time.Instant;
import java.util.Base64;
import java.util.UUID;

/**
 * One student's request to see one room. The id is the student's private page (/room/{id}). The landlord gets a
 * separate token (/film/{token}) that can send a video but can't open the student's page. The same landlord link
 * takes a new video any time, so when the student asks for a better one, the landlord just films again.
 */
@Entity
public class RoomRequest {

    private static final SecureRandom RANDOM = new SecureRandom();

    @Id
    private String id;

    @Column(unique = true, nullable = false)
    private String landlordToken;

    /** What the student calls the place, e.g. "Room on Champlain Cres". */
    private String label;

    @Enumerated(EnumType.STRING)
    private RequestStatus status;

    private Instant createdAt;
    private Instant videoReceivedAt;
    private Instant readyAt;

    @Enumerated(EnumType.STRING)
    private VideoSource videoSource;
    private String videoFile;

    // The current 3D run, and what the helper says it's doing right now
    private String model;
    private String workerJobId;
    private String stage;

    // What the last finished run produced (see RoomStats)
    private Integer frames;
    private Double seconds;
    private Double peakVramGb;
    private Double scaleFactor;
    private Long points;
    private Double sizeX;
    private Double sizeY;
    private Double sizeZ;

    @Column(length = 2000)
    private String failureMessage;

    protected RoomRequest() {
    }

    public static RoomRequest create(String label) {
        RoomRequest r = new RoomRequest();
        r.id = UUID.randomUUID().toString();
        r.landlordToken = newToken();
        r.label = label;
        r.status = RequestStatus.WAITING_FOR_VIDEO;
        r.createdAt = Instant.now();
        return r;
    }

    private static String newToken() {
        byte[] bytes = new byte[18];
        RANDOM.nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    void videoArrived(VideoSource source, String fileName) {
        videoSource = source;
        videoFile = fileName;
        videoReceivedAt = Instant.now();
        status = RequestStatus.VIDEO_RECEIVED;
        stage = null;
        failureMessage = null;
    }

    public void reconstructionStarted(String jobId, String model) {
        this.workerJobId = jobId;
        this.model = model;
        this.status = RequestStatus.RECONSTRUCTING;
        this.stage = "waiting for the GPU";
        this.failureMessage = null;
    }

    /** Returns true when the stage changed, so the caller only saves when there's news. */
    public boolean stageIs(String newStage) {
        if (newStage == null || newStage.equals(stage)) {
            return false;
        }
        stage = newStage;
        return true;
    }

    public void reconstructionDone(RoomStats stats) {
        frames = stats.frames();
        seconds = stats.seconds();
        peakVramGb = stats.peakVramGb();
        scaleFactor = stats.scaleFactor();
        points = stats.points();
        sizeX = stats.sizeX();
        sizeY = stats.sizeY();
        sizeZ = stats.sizeZ();
        status = RequestStatus.READY;
        stage = null;
        readyAt = Instant.now();
    }

    public void failed(String message) {
        status = RequestStatus.FAILED;
        stage = null;
        failureMessage = message;
    }

    public boolean hasVideo() {
        return videoFile != null;
    }

    public boolean hasRoom() {
        return frames != null;
    }

    public String getId() {
        return id;
    }

    public String getLandlordToken() {
        return landlordToken;
    }

    public String getLabel() {
        return label;
    }

    public RequestStatus getStatus() {
        return status;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getVideoReceivedAt() {
        return videoReceivedAt;
    }

    public Instant getReadyAt() {
        return readyAt;
    }

    public VideoSource getVideoSource() {
        return videoSource;
    }

    public String getVideoFile() {
        return videoFile;
    }

    public String getModel() {
        return model;
    }

    public String getWorkerJobId() {
        return workerJobId;
    }

    public String getStage() {
        return stage;
    }

    public Integer getFrames() {
        return frames;
    }

    public Double getSeconds() {
        return seconds;
    }

    public Double getPeakVramGb() {
        return peakVramGb;
    }

    public Double getScaleFactor() {
        return scaleFactor;
    }

    public Long getPoints() {
        return points;
    }

    public Double getSizeX() {
        return sizeX;
    }

    public Double getSizeY() {
        return sizeY;
    }

    public Double getSizeZ() {
        return sizeZ;
    }

    public String getFailureMessage() {
        return failureMessage;
    }
}
