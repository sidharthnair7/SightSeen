package dev.sightseen.reconstruction;

import dev.sightseen.SightSeenProperties;
import dev.sightseen.request.RequestFiles;
import dev.sightseen.request.RequestStatus;
import dev.sightseen.request.RoomRequest;
import dev.sightseen.request.RoomRequestRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClientException;

import java.nio.file.Path;
import java.util.Optional;

/**
 * Hands videos to the 3D helper and keeps each request up to date while the helper works. The helper runs one job
 * at a time on the GPU, so a run can take minutes; the student's page just shows the stage the helper reports.
 */
@Service
public class ReconstructionService {

    private static final Logger log = LoggerFactory.getLogger(ReconstructionService.class);

    private final RoomRequestRepository requests;
    private final RequestFiles files;
    private final WorkerClient worker;
    private final SightSeenProperties.Worker settings;

    public ReconstructionService(RoomRequestRepository requests, RequestFiles files, WorkerClient worker,
                                 SightSeenProperties props) {
        this.requests = requests;
        this.files = files;
        this.worker = worker;
        this.settings = props.worker();
    }

    /** Sends the request's video to the helper with the given model, or the usual one when it's blank. */
    public void start(RoomRequest r, String model) {
        String chosen = (model == null || model.isBlank()) ? settings.model() : model;
        Path folder = files.folder(r.getId());
        String jobId = r.getId() + "-" + System.currentTimeMillis();
        try {
            worker.submit(new WorkerClient.JobRequest(jobId, folder.resolve(r.getVideoFile()).toString(),
                    folder.toString(), chosen, settings.fps(), settings.maxFrames()));
            r.reconstructionStarted(jobId, chosen);
        } catch (HttpClientErrorException e) {
            r.failed("The 3D helper turned the job down: " + e.getResponseBodyAsString());
        } catch (RestClientException e) {
            log.warn("3D helper unreachable at {}: {}", settings.url(), e.getMessage());
            r.failed("The 3D helper isn't running. Start it (worker/run.ps1), then press Try again.");
        }
        requests.save(r);
    }

    /** Every few seconds, asks the helper how each running job is doing. */
    @Scheduled(fixedDelay = 3000)
    public void poll() {
        for (RoomRequest running : requests.findByStatus(RequestStatus.RECONSTRUCTING)) {
            String jobId = running.getWorkerJobId();
            Optional<WorkerClient.JobStatus> status;
            try {
                status = worker.status(jobId);
            } catch (RestClientException e) {
                // The helper is down for a moment. Keep waiting; if it comes back without the job, the next poll says so.
                log.debug("3D helper not answering: {}", e.getMessage());
                continue;
            }
            // Re-read the request: a newer video may have replaced this job while we were asking.
            requests.findById(running.getId())
                    .filter(r -> r.getStatus() == RequestStatus.RECONSTRUCTING && jobId.equals(r.getWorkerJobId()))
                    .ifPresent(r -> {
                        if (apply(r, status)) {
                            requests.save(r);
                        }
                    });
        }
    }

    /** Returns true when the request changed. */
    private static boolean apply(RoomRequest r, Optional<WorkerClient.JobStatus> status) {
        if (status.isEmpty()) {
            r.failed("The 3D helper restarted and lost this job. Press Try again.");
            return true;
        }
        WorkerClient.JobStatus s = status.get();
        switch (s.state()) {
            case "done" -> {
                if (s.result() == null) {
                    r.failed("The 3D helper finished without a room.");
                } else {
                    r.reconstructionDone(s.result());
                }
                return true;
            }
            case "failed" -> {
                r.failed(s.message());
                return true;
            }
            default -> {
                return r.stageIs(s.stage());
            }
        }
    }
}
