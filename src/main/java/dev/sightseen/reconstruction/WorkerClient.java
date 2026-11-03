package dev.sightseen.reconstruction;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import dev.sightseen.SightSeenProperties;
import dev.sightseen.request.RoomStats;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.Optional;

/** Talks to the Python helper (worker/app.py) that runs MapAnything. */
@Component
public class WorkerClient {

    /** A job for the helper. The paths are on this machine; the helper reads the video and writes next to it. */
    public record JobRequest(String jobId, String videoPath, String outputDir, String model, double fps,
                             int maxFrames) {
    }

    /** One of the student's things to build from its photo (worker/items.py): items/{itemId}/ in the room's folder. */
    public record ItemJob(String itemId, String requestDir) {
    }

    /**
     * A real length for the room: two points measured in the viewer (room units apart) and what a tape says they are.
     * The helper sets the room's scale from it and makes again everything the scale sizes.
     */
    public record ScaleRequest(String requestDir, double units, double metres) {
    }

    /** The helper's answer about a job: state is queued, running, done or failed. */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record JobStatus(String jobId, String state, String stage, String message, RoomStats result) {
    }

    private final RestClient http;
    private final RestClient slow; // the scale: the walls' photos and materials are made again, which takes a minute

    public WorkerClient(SightSeenProperties props) {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(2))
                .build();
        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(client);
        factory.setReadTimeout(Duration.ofSeconds(15));
        this.http = RestClient.builder().baseUrl(props.worker().url()).requestFactory(factory).build();
        JdkClientHttpRequestFactory patient = new JdkClientHttpRequestFactory(client);
        patient.setReadTimeout(Duration.ofMinutes(4));
        this.slow = RestClient.builder().baseUrl(props.worker().url()).requestFactory(patient).build();
    }

    /** Sets the room's scale from a real length (worker/room_scale.py); the helper's JSON answer as it is. */
    public String setScale(ScaleRequest scale) {
        return slow.post().uri("/scale")
                .contentType(MediaType.APPLICATION_JSON)
                .body(scale)
                .retrieve()
                .body(String.class);
    }

    public JobStatus submit(JobRequest job) {
        return http.post().uri("/jobs")
                .contentType(MediaType.APPLICATION_JSON)
                .body(job)
                .retrieve()
                .body(JobStatus.class);
    }

    /** Queues a thing's build; its progress is in its own item.json, which the helper writes. */
    public void submitItem(ItemJob job) {
        http.post().uri("/items")
                .contentType(MediaType.APPLICATION_JSON)
                .body(job)
                .retrieve()
                .toBodilessEntity();
    }

    /** Empty when the helper doesn't know the job, which happens after it restarts. */
    public Optional<JobStatus> status(String jobId) {
        try {
            return Optional.ofNullable(http.get().uri("/jobs/{id}", jobId).retrieve().body(JobStatus.class));
        } catch (HttpClientErrorException.NotFound e) {
            return Optional.empty();
        }
    }
}
